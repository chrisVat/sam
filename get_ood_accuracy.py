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
import answer_extract_utils 
from transformers import GenerationConfig
import prompt_utils

VAL = True
FORM = 'step'
IGNORE = -100
SHOTS = 0
FLAN = ""

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




def exact_match_accuracy(
        data,
        model,
        tokenizer,
        batch_size: int = 4,
        max_new_tokens: int = 128,
        debug: bool = False,
        source: str = "train_dataset",
        max_length: int = 300,
        do_sample: bool = False,
    ):

    # set padding side to left for the tokenizer
    tokenizer.padding_side = "left"

    model.cuda()
    model.eval()

    ds = data[source]
    collator = data["data_collator"]

    loader = DataLoader(
        ds,
        batch_sampler=LengthSortedBatchSampler(ds, batch_size),
        collate_fn=collator,
        drop_last=False,
    )

    hits, total = 0, 0
    with torch.no_grad():
        progress = tqdm(loader, desc="EM")
        for batch in progress:
            input_ids = batch["input_ids"].cuda()
            #if "labels" in batch:
            attention_mask = batch["attention_mask"].cuda() if "attention_mask" in batch else None

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                generation_config=GenerationConfig(
                    do_sample=do_sample, 
                    max_new_tokens=max_length, 
                    trust_remote_code=True)
                )
            output_strs = []

            for output_id in output_ids.tolist():
                tmp = tokenizer.decode(output_id[input_ids.shape[-1]:], skip_special_tokens=True)
                output_strs.append(tmp)

            # convert labels to ground truth
            groundtruths = []

            labels_clean = batch["labels"].clone()
            labels_clean[labels_clean == -100] = tokenizer.pad_token_id
            groundtruths = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

            
            questions = []
            for input_id in input_ids.tolist():
                tmp = tokenizer.decode(input_id, skip_special_tokens=True)
                questions.append(tmp)


            #print("Outputs:", output_strs)
            #print("Groundtruths:", groundtruths)
            #print("Questions:", questions)
            #exit()

            returned_value = []
            rerun_questions = []
            rerun_groundtruths = []


            for output, question, groundtruth in zip(output_strs, questions, groundtruths):
                #print(f"Output: {output}")
                #print(f"Question: {question}")
                #print(f"Groundtruth: {groundtruth}")
                
                if 'print(' in output:
                    output = output.split("### Instruction")[0]
                    tmp = answer_extract_utils.execute_with_timeout(output)
                    tmp = 'The answer is' + ' ' + tmp
                    answer = answer_extract_utils.answer_clean("math", ('####', 'The answer is'), tmp)
                else:
                    answer = answer_extract_utils.answer_clean("math", ('####', 'The answer is'), output)

                ground_truth_answer = answer_extract_utils.answer_clean("math", ('####', 'The answer is'), groundtruth)
                print(f"Answer: {answer}")
                print(f"Ground Truth Answer: {ground_truth_answer}")

                hits += int(answer == ground_truth_answer)
                total += 1



            progress.set_postfix(acc=hits / max(total, 1))

    return hits / max(total, 1)

def main(model_path, config_file=None, ckpt=-1, dataset_name=None):
    path_friendly = dataset_name.replace("/", "__") if dataset_name else "default_dataset"
    accuracy_file_name = f"{path_friendly}_accuracy.pt" if not VAL else f"{path_friendly}_val_accuracy.pt"

    if config_file:
        # Local model path logic
        with open(config_file, 'r') as f:
            args = yaml.full_load(f)
        
        args["full_data_path"] = dataset_name if dataset_name else args.get("full_data_path", None)
        
        rank0_print('Configuration loaded!')
        rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

        args["data_path_root"] = f"res/data"
        args["output_dir_root"] = f"res/{args['result_dir_name']}/output"
        
        if ckpt == -1:
            model_path = args["output_dir_root"]+f"/"
        else:
            model_path = args["output_dir_root"]+f"/checkpoint-{ckpt}"

        accuracy_file = f"{model_path}/{accuracy_file_name}" 
    else:
        # HuggingFace model path logic
        args = {
            "cache_dir": None,
            #"model_max_length": 2048,  # You might want to make this configurable
            "model_max_length": 900,  # You might want to make this configurable
            "model_name_or_path": model_path,
            "full_data_path": dataset_name if dataset_name else None,
            "data_path_root": "res/data",
        }
        if ckpt != -1:
            model_path = f"{model_path}@{ckpt}"
        
        # Create a default output directory for HF models
        os.makedirs("hf_outputs", exist_ok=True)
        accuracy_file = f"hf_outputs/{model_path.replace('/', '_')}_{accuracy_file_name}" 

    if os.path.exists(accuracy_file):
        rank0_print(f"***** Accuracies already exist at {accuracy_file}!")
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
    
    print("data path: ", args.get("full_data_path", None))
    #exit()
    
    all_data = make_supervised_data_module(tokenizer=tokenizer, data_path=args["full_data_path"])
    print(f"***** Data loaded from {args['full_data_path']}!")
    #exit()

    #exit()


    accuracy = exact_match_accuracy(
        data=all_data,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        debug=True,      # set True to print pred/gold pairs
    )
    print(f"Exact‑match accuracy: {accuracy:.4%}")

    # load file (it exists)
    #old_mean_entropies_all = torch.load(loss_file)
    # convert from float to tensor
    #difference = torch.stack(mean_entropies_all) - torch.stack(old_mean_entropies_all)
    #new_loss_file = loss_file.replace(".pt", "_new.pt")
    #torch.save(mean_entropies_all, new_loss_file)
    #print(f"***** Difference in losses: {difference.mean().item():.6f}")
    #exit()

    torch.save(accuracy, accuracy_file)
    print(f"***** Accuracies saved to {accuracy_file}")
    return accuracy


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
    start_ckpt = 44000
    
    start_ckpt = 36500
    
    #start_ckpt = 2500

    # include rSVAMP, Mathematics, SimulEq
    datasets = [
        #("ChilleD/SVAMP",            None),
        #("deepmind/math_dataset",    None),
        #("simuleq",             None),
        ("gsm8k", "main"),       # or "hard"
        #("EleutherAI/hendrycks_math", None),          # Hendrycks et al. MATH
        #("numglue", None),       # Mishra et al.
        #("TIGER-Lab/MathInstruct", None),  # MathInstruct dataset
    ]

    #datasets = [
    #    ("TiGER-Lab/MathInstruct", None),  # MathInstruct dataset
    #]

    config_files = [
        #"./configs/small-proxy-full.yml",
        #"./configs/s2l_relative_upsample_15_full.yml",
        
        #'./configs/phi2-full-lora.yml',
        #'./configs/s2l_relative_upsample_15_full-phi2-lora.yml',
        './configs/s2l_relative_upsample_6_full-phi2-lora.yml',
    ]

    ckpt_nums = [
        #5835, 
        #8214, 
        #8205,
        8000
    ]




    for config_idx, config_file in enumerate(config_files):
        if len(ckpt_nums) > 0:
            args.ckpt = ckpt_nums[config_idx]
        else:
            args.ckpt = start_ckpt
        args.config_file = config_file
        for dataset_name in datasets:
            cur_name = dataset_name[0] if dataset_name[1] is None else f"{dataset_name[0]}:{dataset_name[1]}"
            print(cur_name)
            args.full_data_path = dataset_name
            accuracy = main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt, dataset_name=cur_name)
            if accuracy is not None:
                print(f"***** Accuracy for {cur_name} at ckpt {args.ckpt}: {accuracy:.6f}")

    #main(model_path=args.model_path, config_file=args.config_file, ckpt=args.ckpt)


    # CUDA_VISIBLE_DEVICES=0 python get_trajectories.py