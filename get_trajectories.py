import argparse
import torch
import yaml
from tqdm import tqdm
from utils import rank0_print, get_model, get_tokenizer, smart_tokenizer_and_embedding_resize, make_supervised_data_module
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Sampler, DataLoader

VAL = False


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
    """Compute per-example loss with batching and compare to model's result.loss.
    This version manually shifts the logits/labels to match the model's internal loss computation.
    """
    model.cuda()
    model.eval()
    
    #losses = []
    losses = {}
    collator = data["data_collator"]
    source = "train_dataset" #if not VAL else "eval_dataset"
    print("data keys: ", data.keys())

    eval_dataset = data[source]
    print("eval dataset: ", eval_dataset)
    print("len eval dataset: ", len(eval_dataset))
    #exit()

    dataloader = DataLoader(data[source], 
                            #batch_size=batch_size, 
                            #shuffle=False, 
                            collate_fn=collator,
                            batch_sampler=LengthSortedBatchSampler(data[source], batch_size)
                            )
    
    #n = 16
    #run_first_n_losses(data, model, n)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            example_ids = batch["id"]

            result = model(input_ids=input_ids, labels=labels, return_dict=True)
            #batch_loss = result.loss  # Scalar loss from the model
            
            logits = result.logits 
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
            per_token_loss = per_token_loss.view(shifted_labels.shape)
            
            valid_mask = shifted_labels != -100
            per_example_losses = per_token_loss.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            
            # extend losses 
            losses.update({example_id: per_example_loss.detach().cpu() for example_id, per_example_loss in zip(example_ids, per_example_losses)})

            #for i, example_id in enumerate(example_ids):
            #    losses[example_id] = per_example_losses[i].detach().cpu() #.item()
            
            
            #losses.extend(per_example_losses.detach().cpu())

            #for per_example_loss in per_example_losses:
            #    losses.append(per_example_loss.detach().cpu())
    
    # convert losses to a list
    losses = [losses[k] for k in sorted(losses.keys())]
    
    return losses


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
        return # add me back!
    
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
    model = get_model(model_name_or_path=model_path, cache_dir=args["cache_dir"])
    rank0_print(f'***** Model loaded from {model_path}!') 
    tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"], model_max_length=args["model_max_length"],)
    rank0_print(f'***** Tokenizer initilized!')
    tokenizer, model = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                            tokenizer=tokenizer, 
                                                            model=model)  # fix tokenizer's special_token_maps
    rank0_print(f'***** smart_tokenizer_and_embedding_resize done!')
    
    
    all_data = make_supervised_data_module(tokenizer=tokenizer, data_path=args["full_data_path"])

    mean_entropies_all = loss(data=all_data, model=model)

    # load file (it exists)
    #old_mean_entropies_all = torch.load(loss_file)
    # convert from float to tensor
    #difference = torch.stack(mean_entropies_all) - torch.stack(old_mean_entropies_all)
    #new_loss_file = loss_file.replace(".pt", "_new.pt")
    #torch.save(mean_entropies_all, new_loss_file)
    #print(f"***** Difference in losses: {difference.mean().item():.6f}")
    #exit()


    torch.save(mean_entropies_all, loss_file)
    print(f"***** Losses saved to {loss_file}")      
      

def loss_old(data, model):
    """compute last hidden states for a data_module"""
    model.cuda()
    model.eval()
    
    losses = []
    
    source = "train_dataset" if not VAL else "eval_dataset"

    with torch.no_grad():
        for _,datapoint in tqdm(enumerate(data[source]), total=len(data[source])):
            input_ids = datapoint["input_ids"].unsqueeze(0).cuda()
            labels = datapoint["labels"].unsqueeze(0).cuda()
            result = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = result.loss
            if _==1 or (_!=0 and _%10000 == 0): # report progress
                rank0_print(f"***** Predict-Progress -- {_} DONE !")
            losses.append(loss.detach().cpu())
    return losses

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

    args.config_file = './configs/small-proxy.yml'
    #args.config_file = './configs/s2l-preconfsam_05-singlegpu_2e-5.yml'

    #args.ckpt = 5000
    args.ckpt = 12000
    args.model_path = None 

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)
    #exit()

    inc = 500
    start_ckpt = 500
    while start_ckpt <= 25000:
        args.ckpt = start_ckpt
        main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)
        start_ckpt += inc

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)


    # CUDA_VISIBLE_DEVICES=0 python get_trajectories.py