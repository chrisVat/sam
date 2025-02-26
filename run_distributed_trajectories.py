import subprocess
import torch
import time
from itertools import cycle
import argparse
import os
import re

USE_GPU = -1

def get_available_gpus():
    """Returns list of available GPU indices"""
    return list(range(torch.cuda.device_count()))

def get_checkpoint_list(model_path):
    """Get list of available checkpoint numbers from the model path"""
    print(model_path)
    checkpoints = []
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    
    for item in os.listdir(model_path):
        match = checkpoint_pattern.match(item)
        if match and os.path.isdir(os.path.join(model_path, item)):
            checkpoints.append(int(match.group(1)))
    
    return sorted(checkpoints)

def run_distributed_checkpoints(model_path, config_file, checkpoint_list):
    # Get available GPUs
    gpus = get_available_gpus()
    if USE_GPU != -1:
        gpus = [USE_GPU]
    
    if not gpus:
        raise RuntimeError("No GPUs available!")
    
    print(f"Found {len(gpus)} GPUs: {gpus}")
    
    # Create a cycle of available GPUs
    gpu_cycle = cycle(gpus)
    
    # Track running processes
    running_tasks = {}  # {process: (gpu_id, checkpoint)}
    
    # Process all checkpoints
    while checkpoint_list or running_tasks:
        # Start new processes if GPUs are available and there are checkpoints to process
        while checkpoint_list and len(running_tasks) < len(gpus):
            gpu_id = next(gpu_cycle)
            ckpt = checkpoint_list.pop(0)
            #if USE_GPU != -1:
            #    gpu_id=USE_GPU
            
            cmd = [
                "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
                "python", "get_trajectories.py",
                "--model_path", model_path,
            ]
            print(cmd)
            if config_file:
                cmd.extend(["--config_file", config_file])
            
            cmd.extend(["--ckpt", str(ckpt)])
            
            print(f"Starting checkpoint {ckpt} on GPU {gpu_id}")
            process = subprocess.Popen(" ".join(cmd), shell=True)
            running_tasks[process] = (gpu_id, ckpt)
        
        # Check for completed processes
        for process in list(running_tasks.keys()):
            if process.poll() is not None:  # Process has finished
                gpu_id, ckpt = running_tasks[process]
                if process.returncode == 0:
                    print(f"Checkpoint {ckpt} completed successfully on GPU {gpu_id}")
                else:
                    print(f"Checkpoint {ckpt} failed on GPU {gpu_id} with return code {process.returncode}")
                del running_tasks[process]
                #exit()
        
        # Small sleep to prevent CPU overload
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Either local model directory (with config) or HuggingFace model path')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file path for local models')
    parser.add_argument('--checkpoints', type=str, default='all',
                        help='Comma-separated list of checkpoint numbers, range in format start:end:step, or "all" to process all checkpoints')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use for processing')
    

    args = parser.parse_args()
    USE_GPU = args.gpu
    
    # Parse checkpoint list
    if args.checkpoints.lower() == 'all':
        checkpoint_list = get_checkpoint_list(args.model_path)
    elif ':' in args.checkpoints:
        start, end, step = map(int, args.checkpoints.split(':'))
        checkpoint_list = list(range(start, end + 1, step))
    else:
        checkpoint_list = [int(x) for x in args.checkpoints.split(',')]
    

    max_checkpoint = 3500
    every_n = 2
    remove_first = False
    checkpoint_list = [x for x in checkpoint_list if x <= max_checkpoint]
    
    if remove_first:
        checkpoint_list = checkpoint_list[1:]
    checkpoint_list = checkpoint_list[::every_n]

    checkpoint_list = [4000, 1000, 1500, 2000, 2500, 3000, 3500]
    print(f"Found {len(checkpoint_list)} checkpoints: {checkpoint_list}")

    run_distributed_checkpoints(args.model_path, args.config_file, checkpoint_list) 
    
# Process all checkpoints found in the model directory:
# python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints all

# python run_distributed_trajectories.py --model_path res/fsam-0.05-pythia-70m-deduped-100-oneshot-130k_mathinstruct_phi-2_3epochs_512_4checkpoints/output/ --config_file configs/prefsam_05-train-mathinstruct4memtest.yml --checkpoints all --gpu 3
# python run_distributed_trajectories.py --model_path res/default-pythia-70m-deduped-100-oneshot-130k_mathinstruct_phi-2_3epochs_512_final/output/ --config_file configs/default-train-mathinstruct-final.yml --checkpoints all --gpu 2


# For a comma-separated list of checkpoints:
# python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints 1000,2000,3000,4000

# Or using a range:
# python run_distributed_trajectories.py --model_path /path/to/model --config_file config.yaml --checkpoints 1000:5000:1000