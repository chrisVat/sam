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


def main(config_file):
    with open(config_file, 'r') as f:
        args = yaml.full_load(f)
    loss_file_name = "val_losses.pt" if VAL else "losses.pt"
    
    args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
    # iterate to get all of the folders in their, sort them on checkpoint-[NUMBER]

    #print("output dir root:", args["output_dir_root"])
    #print(os.listdir(args["output_dir_root"]))

    folders = [f for f in os.listdir(args["output_dir_root"]) if f.startswith("checkpoint-") and os.path.isdir(os.path.join(args["output_dir_root"], f))]
    folders.sort(key=lambda x: int(x.split("-")[1]))
    
    all_losses = []
    checkpoints = []

    for folder in tqdm(folders, desc="Processing checkpoints"):
        # load folder-val_losses.pt
        if os.path.exists(f"{args['output_dir_root']}/{folder}/{loss_file_name}"):
            loss_file = f"{args['output_dir_root']}/{folder}/{loss_file_name}"
            if not os.path.exists(loss_file):
                print(f"Loss file {loss_file} does not exist, skipping...")
                continue
            
            losses = torch.tensor(torch.load(loss_file))
            token_count_file = loss_file.replace(".pt", "_token_counts.pt")
            token_counts = torch.load(token_count_file) if os.path.exists(token_count_file) else None

            losses = np.array([l.item() for l in losses])
            token_counts = np.array(token_counts)
            average_loss = np.sum(losses * token_counts) / np.sum(token_counts)

            all_losses.append(average_loss)
            print("Checkpoint:", int(folder.split("-")[1]), "Loss:", average_loss)
            checkpoints.append(int(folder.split("-")[1]))
    
    return all_losses, checkpoints


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

    args.config_file = './configs/small-proxy-full.yml'
    #args.config_file = './configs/s2l-preconfsam_05-singlegpu_2e-5.yml'

    #args.ckpt = 5000
    args.ckpt = 12000
    args.model_path = None 

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)
    #exit()

    inc = 500
    start_ckpt = 38500
    #start_ckpt = 2500

    # include rSVAMP, Mathematics, SimulEq
    datasets = [
        #("ChilleD/SVAMP",            None),
        #("deepmind/math_dataset",    None),
        #("simuleq",             None),
        #("gsm8k", "main"),       # or "hard"
        #("EleutherAI/hendrycks_math", None),          # Hendrycks et al. MATH
        #("numglue", None),       # Mishra et al.
        # mathinstruct
        ("TIGER-Lab/MathInstruct", None),  # MathInstruct dataset
    ]

    config_files = [
        #"./configs/small-proxy-full.yml",
        #"./configs/s2l_relative_upsample_15_full.yml",
        './configs/phi2-full-lora.yml',
        './configs/s2l_relative_upsample_6_full-phi2-lora.yml',
        './configs/s2l_relative_upsample_15_full-phi2-lora.yml',
    ]



    import matplotlib.pyplot as plt
    run_anyway = True

    for config_file in config_files:
        loss_file = f'loss_curves/losses_{config_file.split("/")[-1].split(".")[0]}.pt'
        checkpoint_file = f'loss_curves/checkpoints_{config_file.split("/")[-1].split(".")[0]}.pt'
        if not os.path.exists(loss_file) or run_anyway:
            losses, checkpoints = main(config_file=config_file)
            if not os.path.exists('loss_curves'):
                os.makedirs('loss_curves')
            
            torch.save(losses, loss_file)
            torch.save(checkpoints, checkpoint_file)
        else:
            losses = torch.load(loss_file)
            checkpoints = torch.load(checkpoint_file)

        # identify lowest loss checkopint
        min_loss = min(losses)
        min_loss_index = losses.index(min_loss)
        min_checkpoint = checkpoints[min_loss_index]
        print(f"Lowest loss: {min_loss:.6f} at checkpoint {min_checkpoint}")

        plt.figure(figsize=(10, 5))
        plt.plot(checkpoints, losses, marker='o', label='Average Loss')
        plt.xlabel('Checkpoint')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per Checkpoint')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_curves/loss_plot_{config_file.split("/")[-1].split(".")[0]}.png')

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)


    # CUDA_VISIBLE_DEVICES=0 python get_trajectories.py