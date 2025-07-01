import argparse
import os
import yaml

from schedules import Full, S2L, S2LUpsample
from utils import get_tokenizer, smart_tokenizer_and_embedding_resize, get_model, rank0_print
import torch.distributed as dist
import datetime
import torch
from utils import is_running_distributed

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29511 train.py --config_file configs/a1_debug-run-s2lupsample.yml --wandb_key $WANDB_KEY > s2lup_0.txt
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29512 train.py --config_file configs/a2_debug-run-s2lupsample.yml --wandb_key $WANDB_KEY > s2lup_1.txt
# CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29513 train.py --config_file configs/a3_debug-run-s2lupsample.yml --wandb_key $WANDB_KEY > s2lup_2.txt
# CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29514 train.py --config_file configs/a4_debug-run-s2lupsample.yml --wandb_key $WANDB_KEY > s2lup_3.txt


ESTABLISH_KILLSWITCH = False
if ESTABLISH_KILLSWITCH:
    from killswitch import setup_killswitch
    setup_killswitch()


if is_running_distributed():
    if not dist.is_initialized():
        timeout_secs = 28800*2
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=timeout_secs))
        torch.cuda.set_device(dist.get_rank())
        print(f"Initialized process group with timeout {timeout_secs/3600} hours.")


## GET_SCHEDULES
def get_schedule(schedule_name):
    if schedule_name == "Full":
        return Full
    elif schedule_name == "S2L":
        return S2L
    elif schedule_name == "S2LCoLM":
        raise NotImplementedError("S2LCoLM schedule is not implemented yet.")
    elif schedule_name == "S2LUpsample":
        return S2LUpsample
    else:
        raise ValueError(f"Unknown schedule name: {schedule_name}")

    
def set_default_values(args):
    # set default values
    if "ref_model_path" not in args:
        args["ref_model_path"] = None
    if "n_components" not in args:
        args["n_components"] = -1
    if "num_loss_ckpts" not in args:
        args["num_loss_ckpts"] = -1
    if "distance" not in args:
        args["distance"] = 'euclidean'
    if "seed" not in args:
        args["seed"] = 42
        
    return args


## RUN
def main(config_file):
    # load configuration
    with open(config_file, 'r') as f:
        args = yaml.full_load(f)
    rank0_print('Configuration loaded!')
    
    # set default values
    args = set_default_values(args)
    rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    # makedirs    
    args["data_path_root"] = f"res/{args['result_dir_name']}/data"
    args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
    os.makedirs(args["data_path_root"], exist_ok=True)
    os.makedirs(args["output_dir_root"], exist_ok=True)

    # Initialize model and tokenizer
    model = get_model(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"])
    
    rank0_print('*** Model initialized!')
    
    tokenizer, special_tokens_dict = get_tokenizer(
        model_name_or_path=args["model_name_or_path"], 
        cache_dir=args["cache_dir"], 
        model_max_length=args["model_max_length"]
    )
    rank0_print('*** Tokenizer initialized!')
    
    tokenizer, model = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )
    rank0_print('*** Smart tokenizer and embedding resize done!')

    # Initialize schedule
    schedule = get_schedule(schedule_name=args["schedule_name"])(
        model=model,
        tokenizer=tokenizer,
        args=args
    )
    rank0_print('*** Schedule built!')

    # Initialize data
    schedule.initialize_labeled_data()
    
    #print("outer check for per example usage:", schedule.per_example_usages)

    schedule.save_labeled_unlabeled_data()
    rank0_print(f"*** Training-Data-Size = {len(schedule.labeled_idx[schedule.labeled_idx==True])}")
    rank0_print(f"*** Batch Size = {args['train_args']['per_device_train_batch_size'] * args['train_args']['gradient_accumulation_steps']}")

    # Train
    schedule.train()
    rank0_print("*** Training Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,)
    parser.add_argument('--wandb_key', type=str, required=True, help="wandb login key")
    args = parser.parse_args()
    
    import wandb
    wandb.login(key=args.wandb_key)
    
    main(config_file=args.config_file)
