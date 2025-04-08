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


def activations(data, model, batch_size=32):
    """Compute per-example loss with batching and compare to model's result.loss.
    This version manually shifts the logits/labels to match the model's internal loss computation.
    """
    model.cuda()
    model.eval()
    
    #losses = []
    activations = {}
    collator = data["data_collator"]
    dataloader = DataLoader(data["train_dataset"], 
                            #batch_size=batch_size, 
                            #shuffle=False, 
                            collate_fn=collator,
                            batch_sampler=LengthSortedBatchSampler(data["train_dataset"], batch_size)
                            )
    
    #n = 16
    #run_first_n_losses(data, model, n)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            example_ids = batch["id"]

            result = model(input_ids=input_ids, return_dict=True, output_hidden_states=True)

            final_activations = result.hidden_states[-1]

            print(f"***** Activations shape: {final_activations.shape}")
            exit()

            # Save or process activations here
            for example_id, activation in zip(example_ids, final_activations):
                # You could reduce over sequence length (e.g., mean pooling) if you want a per-example vector
                activation_vector = activation.mean(dim=0).detach().cpu()
                activations[example_id] = activation_vector

            #for i, example_id in enumerate(example_ids):
            #    losses[example_id] = per_example_losses[i].detach().cpu() #.item()
            
            
            #losses.extend(per_example_losses.detach().cpu())

            #for per_example_loss in per_example_losses:
            #    losses.append(per_example_loss.detach().cpu())
    
    # convert losses to a list
    activations = [activations[k] for k in sorted(activations.keys())]
    
    return activations


def main(model_path, config_file=None, ckpt=-1):
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
            
        activations_file = f"{model_path}/activations.pt"
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
        activations_file = f"hf_outputs/{model_path.replace('/', '_')}_activations.pt"

    if os.path.exists(activations_file):
        rank0_print(f"***** Activations already exist at {activations_file}!")
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

    activations_all = activations(data=all_data, model=model)

    # load file (it exists)
    #old_mean_entropies_all = torch.load(loss_file)
    # convert from float to tensor
    #difference = torch.stack(mean_entropies_all) - torch.stack(old_mean_entropies_all)
    #new_loss_file = loss_file.replace(".pt", "_new.pt")
    #torch.save(mean_entropies_all, new_loss_file)
    #print(f"***** Difference in losses: {difference.mean().item():.6f}")
    #exit()


    torch.save(activations_all, activations_file)
    print(f"***** Losses saved to {activations_file}")      
      

def loss_old(data, model):
    """compute last hidden states for a data_module"""
    model.cuda()
    model.eval()
    
    losses = []
    
    with torch.no_grad():
        for _,datapoint in tqdm(enumerate(data["train_dataset"]), total=len(data["train_dataset"])):
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
    
    with torch.no_grad():
        for _,datapoint in enumerate(data["train_dataset"]):
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
    
    args.config_file = './configs/default-900-train-mathinstruct.yml'
    #args.config_file = './configs/preconfsam_10-singlegpu_2e-5.yml'
    #args.config_file = './configs/preconfsam_05-singlegpu_2e-5.yml'
    
    args.ckpt = 5000
    args.model_path = None 

    main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)