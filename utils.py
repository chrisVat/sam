import logging
from dataclasses import dataclass
from datasets import load_dataset
import os
from typing import Union, Dict, Sequence
import io
import copy
import json
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data.distributed import DistributedSampler
from consts import *
from collections import OrderedDict
from safetensors.torch import load_file
import warnings
from datasets import load_dataset, get_dataset_config_names
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig


def is_running_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


## ALPACA-STYLE PROMPT: forked from https://github.com/tatsu-lab/stanford_alpaca
class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def tokenize(tokenizer, cutoff_len, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()  # labels = input_ids -> training decoder
    return result

def generate_and_tokenize_prompt(tokenizer, cutoff_len, prompter, train_on_inputs, add_eos_token, data_point):
    full_prompt = prompter.generate_prompt(
        instruction=data_point["instruction"],
        input=data_point["input"],
        label=data_point["output"],
    )
    tokenized_full_prompt = tokenize(tokenizer=tokenizer,
                                     cutoff_len=cutoff_len,
                                     prompt=full_prompt,
                                     add_eos_token=True)  # default
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(tokenizer=tokenizer,
                                        cutoff_len=cutoff_len,
                                        prompt=user_prompt,
                                        add_eos_token=True
                                        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def get_prompter(prompt_template_name):
    prompter = Prompter(prompt_template_name)
    return prompter


## GET & FIX TOKENIZERS
def get_tokenizer(model_name_or_path, cache_dir, model_max_length, ):
    tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                    model_max_length=model_max_length,
                    padding_side="right",
                )
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = LLAMA_DEFAULT_PAD_TOKEN
    special_tokens_dict["eos_token"] = LLAMA_DEFAULT_EOS_TOKEN
    special_tokens_dict["bos_token"] = LLAMA_DEFAULT_BOS_TOKEN
    special_tokens_dict["unk_token"] = LLAMA_DEFAULT_UNK_TOKEN
    # PROBLEM !!! -> fixed in smart_tokenizer_and_embedding_resize
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # FIX --> bos/eos/unk/pad
    # special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return tokenizer, special_tokens_dict

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,  
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)  # fix tokenizer special tokens map
    if model!=None:
        model.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return tokenizer, model

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess_no_tokenize(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Fast version: skip tokenization, preserve original_idx structure."""
    input_ids = [[] for _ in sources]
    labels = [[] for _ in sources]
    original_idx = [[] for _ in sources]

    return dict(input_ids=input_ids, labels=labels, original_idx=original_idx)


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    original_idx = examples_tokenized.get("original_idx", None)
    if original_idx is not None:
        original_idx = torch.tensor(original_idx, dtype=torch.long)
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = LLAMA_IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels, original_idx=original_idx)


## DATASETS / DATALOADER
class SupervisedDataset(Dataset):
    """Dataset for sft."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, eval=False):
        super(SupervisedDataset, self).__init__()
        #print("Supervised Dataset Created with data_path:", data_path)
        logging.warning("Loading data...")
        self.val_data = None
        self.train_og_index = None  # store original index if available

        if isinstance(data_path, list): # if data_path is a list of data dicts
            list_data_dict = data_path
        elif data_path.endswith(".jsonl"):
            list_data_dict = load_jsonl(data_path)
            if 'instruction' not in list_data_dict[0]:
                list_data_dict = [{'instruction':data['input'], 'output':data['output'], 'source':data['source']} for data in list_data_dict]
        elif data_path.endswith(".json"):
            while True:
                try:
                    list_data_dict = jload(data_path)
                    print(f"making supervised_dataset -> jload('{data_path}') SUCESSFUL")
                    break
                except Exception as e:
                    # e_str = str(e)
                    # print(f"making supervised_dataset -> jload('{data_path}') FAILED: {e_str}")
                    continue
        elif 'MathInstruct' in data_path:
            print("utils mathinstruct")
            raw_dataset = load_dataset(data_path)["train"]  # fixed -> for indexing all samples

            raw_dataset = raw_dataset.add_column("original_idx", list(range(len(raw_dataset))))

            dataset = raw_dataset.shuffle(seed=42)

            train_num = int(len(dataset)*0.95)

            train_data = dataset.select(range(train_num))

            val_data = dataset.select(range(train_num, len(dataset)))

            self.train_og_index = torch.tensor(train_data["original_idx"])
            self.val_og_index = torch.tensor(val_data["original_idx"])

            list_data_dict = train_data


            #keep_only, keep_only_train = 500, 500
            #list_data_dict = list_data_dict.select(range(keep_only_train))
            #val_data = val_data.select(range(keep_only))

            if eval:
                list_data_dict = val_data


            print("\n[DEBUG] First 3 formatted examples from MathInstruct:\n")
            for i in range(1):
                ex = list_data_dict[i]
                example = {
                    "instruction": ex["instruction"].strip(),
                    "input": "",  # MathInstruct does not have separate 'input' field
                    "output": ex["output"].strip()
                }
                try:
                    formatted = PROMPT_DICT["prompt_input"].format_map(example) \
                        if example["input"] != "" \
                        else PROMPT_DICT["prompt_no_input"].format_map(example)
                    print(f"Example {i} formatted prompt:\n{formatted}")
                    print(f"Expected Output:\n{example['output']}")
                    print("-" * 40)
                except Exception as e:
                    print(f"Formatting failed for example {i}: {e}")


        elif 'Asclepius' in data_path:
            list_data_dict = load_dataset(data_path)["train"]  # fixesd -> for indexing all samples
            list_data_dict = [{'instruction':data['question'], 'input':data['note'], 'output':data['answer'], 'source':data['task']} for data in list_data_dict]
        
        elif 'ChilleD/SVAMP' in data_path or 'svamp' in data_path.lower():
            data_df = load_dataset(data_path)["train"]
            list_data_dict = []

            print("\n[DEBUG] First 3 formatted examples from SVAMP:\n")
            for i, ex in enumerate(data_df):
                instruction = f"{ex['Body'].strip()} {ex['Question'].strip()}"
                output = str(ex['Answer']).strip()

                example = dict(
                    instruction=instruction,
                    input="",  # Force empty input to trigger prompt_no_input
                    output=output,
                    source='SVAMP'
                )

                # Print prompt to visually confirm formatting
                if i < 1:
                    try:
                        formatted = PROMPT_DICT["prompt_no_input"].format_map(example)
                        print(f"Example {i} formatted prompt:\n{formatted}\n{'-'*40}")
                        print(f"Expected Output:\n{example['output']}")
                    except Exception as e:
                        print(f"Formatting error on example {i}: {e}")

                list_data_dict.append(example)

        elif 'deepmind/math_dataset' in data_path:
            list_data_dict = []
            branches = get_dataset_config_names("deepmind/math_dataset", trust_remote_code=True)

            import ast 
            def safe_decode(x):
                if isinstance(x, bytes):
                    return x.decode("utf-8")
                if isinstance(x, str) and x.startswith("b'") and x.endswith("'"):
                    try:
                        # Convert string literal to bytes, then decode
                        return ast.literal_eval(x).decode("utf-8")
                    except Exception as e:
                        print(f"[DEBUG] Failed to unwrap byte string: {x} ({e})")
                        return x
                return x
                
            total_printed = 0

            for branch in branches:
                print(f"\nLoading DeepMind Math branch: {branch}")
                ds = load_dataset("deepmind/math_dataset", branch, split="test", trust_remote_code=True)

                for i, ex in enumerate(ds):
                    question = safe_decode(ex["question"]).strip()
                    answer = safe_decode(ex["answer"]).strip()

                    example = dict(
                        instruction=question,
                        input="",
                        output=answer,
                        source=f"deepmind_math_{branch}"
                    )

                    if total_printed < 1:
                        total_printed += 1
                        try:
                            formatted = (
                                PROMPT_DICT["prompt_input"].format_map(example)
                                if example.get("input", "") != ""
                                else PROMPT_DICT["prompt_no_input"].format_map(example)
                            )
                            print(f"[DEBUG] Branch: {branch}, Example {i} formatted prompt:\n{formatted}\n----------------------------------------")
                            print(f"Expected Output:\n{example['output']}")
                        except Exception as e:
                            print(f"Formatting failed for branch {branch}, example {i}: {e}")

                    list_data_dict.append(example)

        elif 'simuleq' in data_path.lower():
            dataset = load_dataset("allenai/lila", "simuleq", split="train")
            list_data_dict = []

            print("\n[DEBUG] First 3 formatted examples from SimulEq:\n")
            for i, ex in enumerate(dataset):
                instruction = ex["input"].strip()
                output = ex["output_answer"].strip()

                example = dict(
                    instruction=instruction,
                    input="",  # Always empty, enforce no_input prompt
                    output=output,
                    source="SimulEq"
                )

                if i < 1:
                    try:
                        formatted = PROMPT_DICT["prompt_no_input"].format_map(example)
                        print(f"Example {i} formatted prompt:\n{formatted}\n{'-'*40}")
                        print(f"Expected Output:\n{example['output']}")
                    except Exception as e:
                        print(f"Formatting error on SimulEq example {i}: {e}")

                list_data_dict.append(example)

        elif 'gsm8k' in data_path.lower():
            list_data_dict = []
            gsm_split = data_path.split(":")[1] if ":" in data_path else "main"
            ds = load_dataset("gsm8k", gsm_split, split="test")

            print("\n[DEBUG] First 3 formatted examples from GSM8K:\n")
            for i, ex in enumerate(ds):
                instruction = ex["question"].strip()
                output = ex["answer"].strip()

                example = {
                    'instruction': instruction,
                    'input': "",
                    'output': output,
                    'source': f"gsm8k_{gsm_split}"
                }

                if i < 1:
                    try:
                        formatted = PROMPT_DICT["prompt_no_input"].format_map(example)
                        print(f"Example {i} formatted prompt:\n{formatted}\n{'-'*40}")
                        print(f"Expected Output:\n{example['output']}")
                    except Exception as e:
                        print(f"Formatting error on GSM8K example {i}: {e}")

                list_data_dict.append(example)
        elif "hendrycks_math" in data_path.lower():
            list_data_dict = []
            branches = get_dataset_config_names("EleutherAI/hendrycks_math")

            total_printed = 0

            for branch in branches:
                print(f"Loading Hendrycks Math branch: {branch}")
                ds = load_dataset("EleutherAI/hendrycks_math", branch, split="test")

                for i, ex in enumerate(ds):
                    question = ex["problem"].strip()
                    answer = ex["solution"].strip()

                    example = dict(
                        instruction=question,
                        input="",
                        output=answer,
                        source=f"hendrycks_math_{branch}"
                    )

                    if i < total_printed:  # Limit debug print per branch
                        total_printed += 1
                        try:
                            formatted = PROMPT_DICT["prompt_no_input"].format_map(example)
                            print(f"[DEBUG] Branch: {branch}, Example {i} formatted prompt:\n{formatted}\n{'-'*40}")
                            print(f"Expected Output:\n{example['output']}")
                        except Exception as e:
                            print(f"[ERROR] Formatting failed on branch {branch}, example {i}: {e}")

                    list_data_dict.append(example)
        elif "numglue" in data_path.lower():
            import json

            def load_broken_json_array(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                # Split on "}\n{" and fix edge formatting
                objects = content.strip().replace("}\n{", "}|{").split("|")
                return [
                    json.loads(
                        obj if obj.startswith("{") else "{" + obj if not obj.endswith("}") else obj + "}"
                    )
                    for obj in objects
                ]

            list_data_dict = []
            raw_data = load_broken_json_array("datasets/numglue/test.json")

            print("\n[DEBUG] First 3 formatted examples from NumGLUE:\n")
            for i, ex in enumerate(raw_data):
                question = ex.get("question", "").strip()
                raw_ans = ex.get("answer", ex.get("label", ""))
                answer = str(raw_ans).strip()
                task = ex.get("task", "numglue")

                example = {
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "source": task
                }

                if i < 1:
                    try:
                        formatted = PROMPT_DICT["prompt_no_input"].format_map(example)
                        print(f"Example {i} with prompt_input:\n{formatted}\n{'-'*40}")
                        print(f"Expected Output:\n{example['output']}")
                    except Exception as e:
                        print(f"Example {i} formatting FAILED: {e}")

                list_data_dict.append(example)

        else:
            data_df = load_dataset(data_path)["train"]
            # convert to jsonl
            list_data_dict = []
            for i in range(len(data_df)):
                # parse data_df[i]['conversations'] from str to list
                list_data_dict.append(dict(instruction=data_df[i]['conversations'][0], output=data_df[i]['conversations'][1]))
        
        #exit()
        logging.warning("Formatting inputs...")
        print("Length of data_dict:", len(list_data_dict))
        print("utils: self.train_og_index", self.train_og_index)
        #if len(list_data_dict > )

        if 'mimic' in data_path:
            print("Using RADIOLOGY_PROMPT_DICT for MIMIC-CXR dataset")
            prompt_input, prompt_no_input = RADIOLOGY_PROMPT_DICT["prompt_no_input"], RADIOLOGY_PROMPT_DICT["prompt_no_input"]
        else:
            print("Using PROMPT_DICT for general dataset")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        logging.warning("Tokenizing inputs... This may take some time...")
        
        run_fast = False
        if not run_fast:
            data_dict = preprocess(sources, targets, tokenizer)
        else:
            data_dict = preprocess_no_tokenize(sources, targets, tokenizer)
            # just make a list of numbers for input ids!
        
        self.input_ids = data_dict["input_ids"]
        print("utils: SupervisedDataset: train_og_index:", self.train_og_index)
        self.labels = data_dict["labels"]
        self.ids = list(range(len(self.input_ids)))  # Ensure unique IDs are stored
        self.ids = torch.tensor(self.ids, dtype=torch.long)


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #print("getitem called for id: ", self.ids[i])
        #print(type(self.input_ids[i]), type(self.labels[i]), type(self.ids[i]))
        
        return {
            "input_ids": self.input_ids[i], 
            "labels": self.labels[i], 
            "id": self.ids[i],  # Ensure 'id' is a tensor
            "original_idx": self.train_og_index[i] if self.train_og_index is not None else -1
        }
    
    #def get(self, idx) -> Dict[str, torch.Tensor]:
    #    return self.__getitem__(idx)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for sft."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print("collator claled with instances: ", instances[0].keys())
        input_ids, labels, original_idx = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "original_idx"))
        ids = [instance["id"] for instance in instances]
        original_ids = [instance.get("original_idx", None) for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=LLAMA_IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            id=torch.tensor(ids, dtype=torch.long),
            original_idx=torch.tensor(original_ids, dtype=torch.long),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, eval=False, verbose=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if verbose:
        print("utils make supervised data module with data_path:", data_path)
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, eval=eval)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # check if val_data is not None
    eval_dataset = None

    if verbose:
        print("Train data og index:", train_dataset.train_og_index)

    """
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(train_dataset, shuffle=True)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        train_sampler=sampler,
    )
    """

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def load_ddp_state_dict(model_name_or_path, cache_dir=None):
    # Load the safetensor shards
    checkpoint_files = [os.path.join(model_name_or_path, f) for f in os.listdir(model_name_or_path) if f.endswith(".safetensors")]
    checkpoint_files.sort()
    state_dict = {}
    for ckpt in checkpoint_files:
        print(f"Loading {ckpt}...")
        shard = load_file(ckpt)
        state_dict.update(shard) 

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") 
        new_state_dict[new_key] = v

    return new_state_dict


def load_ddp_checkpoint(model_name_or_path, cache_dir=None):
    # Load the safetensor shards
    checkpoint_files = [os.path.join(model_name_or_path, f) for f in os.listdir(model_name_or_path) if f.endswith(".safetensors")]
    checkpoint_files.sort()
    state_dict = {}
    for ckpt in checkpoint_files:
        print(f"Loading {ckpt}...")
        shard = load_file(ckpt)
        state_dict.update(shard) 

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") 
        new_state_dict[new_key] = v

    logging.getLogger("transformers").setLevel(logging.ERROR)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    print("Loading from state dict...")
    model.load_state_dict(new_state_dict, strict=True)
    return model


## GET LLAMA-MODEL
def get_model(model_name_or_path, cache_dir=None, use_lora=False):   
    
    if "t5" in model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    else:
        # get statedict
        #state_dict = torch.load(model_name_or_path, map_location="cpu")
        print("model name: ", model_name_or_path)

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        #model = load_ddp_checkpoint(model_name_or_path, cache_dir=cache_dir)
    
    if use_lora:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print("LoRA model loaded with config:", config)
        
    
    return model


def load_lora_model(adapter_dir: str, cache_dir: str | None = None):
    """
    Loads a LoRA adapter and attaches it to the correct base model.
    Ensures vocab size matches LoRA training checkpoint.
    """
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)

    # Load tokenizer and get intended vocab size
    tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path, use_fast=True)
    vocab_size_lora = 50299  # from your error log, manually setting it

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Resize token embeddings to match LoRA adapter (if needed)
    if base_model.get_input_embeddings().weight.size(0) != vocab_size_lora:
        print(f"Resizing base model vocab from {base_model.get_input_embeddings().weight.size(0)} to {vocab_size_lora}")
        base_model.resize_token_embeddings(vocab_size_lora)

    # Attach the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    return model, tokenizer


## WHITENINING
def compute_kernel_bias(batch_hidden_states):
    """for final transformation: y = (x + bias).dot(kernel)
    batched_hidden_states .shape = (batch_size, hidden_dim)
    """
    mu = batch_hidden_states.mean(axis=0, keepdims=True)  # (1, hidden_dim)
    cov = torch.cov(batch_hidden_states.t())  # (hidden_dim, hidden_dim)
    u, s, vh = torch.linalg.svd(cov)  # u.shape = (hidden_dim, hidden_dim)  s.shape = (hidden_dim)  vh.shape = (hidden_dim, hidden_dim)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))  # (hidden_dim, hidden_dim)
    # kernel = W  # (hidden_dim, hidden_dim)
    # bias = -mu  # (batch_size, hidden_dim)
    return W, -mu

def normalize(batch_hidden_states):
    return batch_hidden_states / (batch_hidden_states**2).sum(dim=1, keepdims=True)**0.5

def transform_and_normalize(batch_hidden_states, kernel, bias):
    """apply transformation & normalization
    batched_hidden_states .shape = (batch_size, hidden_dim)
    kernel .shape = (hidden_dim, hidden_dim) --> 取N_COMPONENTS后 (emb_dim, n_dim)
    bias .shape = (batch_size, hidden_dim) 
    """
    if not (kernel is None or bias is None):
        transformed_batch_hidden_states = torch.mm((batch_hidden_states + bias), kernel)  # (batch_size, n_dim)
    return normalize(transformed_batch_hidden_states)  # (batch_size, n_dim)


## JSON - LOAD/DUMP: forked from https://github.com/tatsu-lab/stanford_alpaca
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict



## OTHERS
def rank0_print(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank()==0:
            print(message)
        else:
            return
    else:
        print(message)
            

def load_jsonl(file):
    lines = []
    with open(file, "r") as f:
        for line in f.readlines():
            lines.append(json.loads(line))
    return lines

