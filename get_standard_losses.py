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

VAL = True



def main(model_path, config_file=None, ckpt=-1, dataset_name=None, val=True):
    loss_file_name = f"losses.pt" if not val else "val_losses.pt"

    # Local model path logic
    with open(config_file, 'r') as f:
        args = yaml.full_load(f)
    
    args["full_data_path"] = dataset_name if dataset_name else args.get("full_data_path", None)
    
    #rank0_print('Configuration loaded!')
    #rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    args["data_path_root"] = f"res/data"
    args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
    
    if ckpt == -1:
        model_path = args["output_dir_root"]+f"/"
    else:
        model_path = args["output_dir_root"]+f"/checkpoint-{ckpt}"

    loss_file = f"{model_path}/{loss_file_name}" 

    #if os.path.exists(loss_file):
        #rank0_print(f"***** Losses already exist at {loss_file}!")
    losses = torch.load(loss_file)
    token_count_file = loss_file.replace(".pt", "_token_counts.pt")
    token_counts = torch.load(token_count_file) if os.path.exists(token_count_file) else None
    return args["result_dir_name"], losses, token_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default=None,
                        help='Config file path for local models')
    parser.add_argument('--model_path', type=str, required=False,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--ckpt', type=int, default=-1,)
    args = parser.parse_args()
    
    args.ckpt = 12000
    args.model_path = None 

    inc = 500
    
    start_ckpt = 38500
    start_ckpt = 66000
    #start_ckpt = 2500

    config_files = [
        #"./configs/small-proxy-full.yml",
        #"./configs/s2l_relative_upsample_15_full.yml",
        #'./configs/s2l_relative_upsample_15_full-phi2-lora.yml',
        './configs/phi2-full-lora.yml'
    ]

    start_ckpt = 5500

    for config_file in config_files:
        args.ckpt = start_ckpt
        args.config_file = config_file
        result_dir_name, losses, token_counts = main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt, val=VAL)
        losses = np.array([l.item() for l in losses])
        token_counts = np.array(token_counts)
        average_loss = np.sum(losses * token_counts) / np.sum(token_counts)

        print(f"*****{result_dir_name}_{args.ckpt}, {f"VAL={VAL}"}: {average_loss:.6f}")

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)


    # CUDA_VISIBLE_DEVICES=0 python get_trajectories.py