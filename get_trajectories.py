import argparse
import torch
import yaml
from tqdm import tqdm
from utils import rank0_print, get_model, get_tokenizer, smart_tokenizer_and_embedding_resize, make_supervised_data_module, load_lora_model
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Sampler, DataLoader

from transformers import TrainingArguments
from consts import LLAMA_IGNORE_INDEX

VAL = True


class LengthSortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        self.indices.sort(key=lambda i: len(dataset[i]["input_ids"]), reverse=True)
        
    def __iter__(self):
        for start_idx in range(0, len(self.indices), self.batch_size):
            yield self.indices[start_idx : start_idx + self.batch_size]
            
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def loss(data, model, batch_size=32):
    """Compute per-example loss and valid token count, ordered by ID."""
    model.cuda()
    model.eval()

    losses = {}
    token_counts = {}
    collator = data["data_collator"]
    source = "train_dataset"
    ignore_index = -100

    eval_dataset = data[source]
    dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        batch_sampler=LengthSortedBatchSampler(eval_dataset, batch_size),
        drop_last=False
    )

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    total_loss_sum = 0.0
    total_tokens = 0

    with torch.no_grad():
        progress = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in progress:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            example_ids = batch["id"]

            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            logits = outputs.logits

            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            valid_mask = shifted_labels != ignore_index

            per_token_loss = ce_loss_fn(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1)
            ).view(shifted_labels.shape)

            batch_loss_sum = (per_token_loss * valid_mask).sum().item()
            batch_tokens = valid_mask.sum().item()
            total_loss_sum += batch_loss_sum
            total_tokens += batch_tokens

            avg_loss = total_loss_sum / max(total_tokens, 1)
            progress.set_postfix(avg_loss=avg_loss)

            # Per-example losses and token counts
            example_loss_sums = per_token_loss.sum(dim=1)
            example_token_counts = valid_mask.sum(dim=1)

            per_example_losses = example_loss_sums / example_token_counts.clamp(min=1)

            for example_id, loss_val, count in zip(example_ids, per_example_losses, example_token_counts):
                losses[example_id] = loss_val.detach().cpu()
                token_counts[example_id] = count.item()

    # Return sorted lists (by example_id)
    ordered_keys = sorted(losses.keys())
    losses = [losses[k] for k in ordered_keys]
    token_counts = [token_counts[k] for k in ordered_keys]
    return losses, token_counts


def main(model_path, config_file=None, ckpt=-1):
    loss_file_name = "losses.pt" if not VAL else "val_losses.pt"
    
    if config_file:
        # Local model path logic
        with open(config_file, 'r') as f:
            args = yaml.full_load(f)
        rank0_print('Configuration loaded!')
        rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

        args["data_path_root"] = f"res/{args['result_dir_name']}/data"
        args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
        
        if ckpt == -1:
            model_path = args["output_dir_root"]+f"/"
        else:
            model_path = args["output_dir_root"]+f"/checkpoint-{ckpt}"

        loss_file = f"{model_path}/{loss_file_name}" 
    else:
        # HuggingFace model path logic
        args = {
            "cache_dir": None,
            #"model_max_length": 2048,  # You might want to make this configurable
            "model_max_length": 900,  # You might want to make this configurable
            "model_name_or_path": model_path
        }
        if ckpt != -1:
            model_path = f"{model_path}@{ckpt}"
        
        # Create a default output directory for HF models
        os.makedirs("hf_outputs", exist_ok=True)
        loss_file = f"hf_outputs/{model_path.replace('/', '_')}_{loss_file_name}" 

    if os.path.exists(loss_file):
        rank0_print(f"***** Losses already exist at {loss_file}!")
        #return # add me back!
    
    """
    new_loss_file = loss_file.replace(".pt", "_new.pt")
    losses = torch.load(loss_file)
    new_losses = torch.load(new_loss_file)
    print(losses)
    print(new_losses)
    exit()
    """


    rank0_print(f"***======================================================================================================")
    rank0_print(f"***** Checkpoint {ckpt} ======================================================================================================")
    
    if args.get("use_lora", False):
        rank0_print(f"***** Using LoRA for model {model_path}!")
        model, tokenizer = load_lora_model(adapter_dir=model_path, cache_dir=args["cache_dir"])
    else:
        model = get_model(model_name_or_path=model_path, cache_dir=args["cache_dir"], use_lora=args.get("use_lora", False),)
    rank0_print(f'***** Model loaded from {model_path}!') 
    tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"], model_max_length=args["model_max_length"],)
    rank0_print(f'***** Tokenizer initilized!')
    tokenizer, model = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                            tokenizer=tokenizer, 
                                                            model=model)  # fix tokenizer's special_token_maps
    rank0_print(f'***** smart_tokenizer_and_embedding_resize done!')
    
    
    all_data = make_supervised_data_module(tokenizer=tokenizer, data_path=args["full_data_path"], eval=VAL)
    # get all data attributes
    # label smoothing factor
    #label_smoothing_factor = args.get("label_smoothing_factor", None)
    #print(f"***** Label smoothing factor: {label_smoothing_factor:.2f}")
        
    #train_args = args["train_args"]
    #train_args = TrainingArguments(**train_args)
    #label_smoothing_factor = train_args.label_smoothing_factor
    #smoother = LabelSmoother(epsilon=label_smoothing_factor, ignore_index=LLAMA_IGNORE_INDEX)

    #print("Train args:", train_args)
    #print("Label smoothing factor:", label_smoothing_factor)

    #exit()

    losses, token_counts = loss(data=all_data, model=model) #, label_smoother=smoother)

    # load file (it exists)
    #old_mean_entropies_all = torch.load(loss_file)
    # convert from float to tensor
    #difference = torch.stack(mean_entropies_all) - torch.stack(old_mean_entropies_all)
    #new_loss_file = loss_file.replace(".pt", "_new.pt")
    #torch.save(mean_entropies_all, new_loss_file)
    #print(f"***** Difference in losses: {difference.mean().item():.6f}")
    #exit()


    torch.save(losses, loss_file)
    torch.save(token_counts, loss_file.replace(".pt", "_token_counts.pt"))
    print(f"***** Losses saved to {loss_file}")      
      

def run_first_n_losses(data, model, n):
    """compute last hidden states for a data_module"""
    model.cuda()
    model.eval()
    
    losses = []
    
    source = "train_dataset" if not VAL else "eval_dataset"
    with torch.no_grad():
        for _,datapoint in enumerate(data[source]):
            input_ids = datapoint["input_ids"].unsqueeze(0).cuda()
            labels = datapoint["labels"].unsqueeze(0).cuda()
            result = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = result.loss
            print(f"Example {len(losses)}: Loss = {loss.item():.6f}")
            losses.append(loss.detach().cpu())
            if len(losses) >= n:
                break
    return losses



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default=None,
                        help='Config file path for local models')
    parser.add_argument('--model_path', type=str, required=False,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--ckpt', type=int, default=-1,)
    args = parser.parse_args()
    
    #args.config_file = './configs/default-900-train-mathinstruct.yml'
    #args.config_file = './configs/preconfsam_10-singlegpu_2e-5.yml'
    #args.config_file = './configs/default-900-train-mathinstruct-sg-mini.yml'
    #args.config_file = './configs/preconfsam_10-singlegpu-mini.yml'

    #args.config_file = './configs/small-proxy-full.yml'
    #args.config_file = './configs/s2l-preconfsam_05-singlegpu_2e-5.yml'
    #args.config_file = './configs/s2l_relative_upsample_15_full.yml'

    #args.ckpt = 5000
    
    args.ckpt = 20000
    args.model_path = None 

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)
    #exit()

    #'s2l_relative_upsample_15_full-phi2-lora.yml',
    #'phi2-full-lora.yml'

    #args.config_file = './configs/phi2-full-lora.yml'
    #args.ckpt = 5835
    #args.ckpt = 1500

    args.config_file = './configs/s2l_relative_upsample_15_full-phi2-lora.yml'
    #args.ckpt = 8214
    #args.ckpt = 1500

    args.config_file = './configs/s2l_relative_upsample_6_full-phi2-lora.yml'
    args.ckpt = 8205
    #args.ckpt = 1500

    #args.ckpt = 2000
    #args.ckpt = 1500
    #args.ckpt = 1000
    #args.ckpt = 500
    inc = 500
    start_ckpt = args.ckpt
    while start_ckpt <= 8500:
        args.ckpt = start_ckpt
        main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)
        start_ckpt += inc
        #break

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)


    # CUDA_VISIBLE_DEVICES=0 python get_trajectories.py