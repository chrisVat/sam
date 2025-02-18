# SAM implementation building off of S2L 
Implementation of functional SAM and SAM designed for accumulation of gradients.  
For now, functional-sam is refered to in code as "prefsam", supposed to be preconditioned functional sam. Hopefully upgraded to that now that functional is working.  

To run:  
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29502 train.py --config_file configs/prefsam-train-mathinstruct.yml --wandb_key $WANDB_KEY > train_prefsam.txt

Running default or SAM = "no" in config may not work right now. 

