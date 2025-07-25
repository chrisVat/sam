import numpy as np
import json
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler
from torch.optim import AdamW
from utils import jload, jdump, make_supervised_data_module, get_model, rank0_print
from sam import SAM
#from functional_sam import PreconditionedFunctionalSAM
from custom_trainer_sam import FSDPSAMTrainer
from custom_trainer_functional_sam import FSDPFunctionalSAMTrainer
from sam_functional import FunctionalSAM 
#from sam_functional_preconditioned import PreconditionedFunctionalSAM
# ddp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from utils import is_running_distributed
from custom_trainer_default import CustomTrainer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

from transformers.models.phi.modeling_phi import PhiDecoderLayer
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler


class Schedule:
    def __init__(self, 
        model, 
        tokenizer,
        args,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.full_data_path = args["full_data_path"]
        self.val_data = None
        self.init_label_num = args["init_label_num"] if "init_label_num" in args else 0
        
        self.load_from = None
        self.load_step = None
        if 'load_from' in args:
            self.load_from = args.pop('load_from')
            self.load_from = f"res/{self.load_from}/output"
            if 'load_step' in args:
                self.load_step = args.pop('load_step')
        
        print("Schedule init, full_data_path:", self.full_data_path)

        # load full-sized source data -> for indexing all samples
        if self.full_data_path.endswith(".jsonl"):
            with open(self.full_data_path, "r") as f:
                self.train_data = [json.loads(line) for line in f]
            val_data_path = self.full_data_path.replace("train", "validate")
            with open(val_data_path, "r") as f:
                self.val_data = [json.loads(line) for line in f]
            self.train_idx = torch.arange(len(self.train_data))
        elif self.full_data_path.endswith(".json"):
            with open(self.full_data_path, "r") as f:
                self.train_data = json.load(f)
        elif "MathInstruct" in self.full_data_path:
            raw_dataset = load_dataset(self.full_data_path)["train"]
            
            raw_dataset = raw_dataset.add_column("original_idx", list(range(len(raw_dataset))))
            
            dataset = raw_dataset.shuffle(seed=42)
            train_num = int(len(dataset)*0.95)
            train_data = dataset.select(range(train_num))
            val_data = dataset.select(range(train_num, len(dataset)))

            #keep_only, keep_only_train = 500, 500
            #train_data = train_data.select(range(keep_only_train))
            #val_data = val_data.select(range(keep_only))
            
            self.train_data = [train_data[i] for i in range(len(train_data))]
            self.val_data = [val_data[i] for i in range(len(val_data))]
            self.train_idx = torch.arange(len(self.train_data))
            self.train_og_index = torch.tensor(train_data["original_idx"])
            #print(self.train_og_index.cpu().tolist())

            #exit()

            self.val_og_index = torch.tensor(val_data["original_idx"])
            self.val_idx = torch.arange(len(self.val_data)) + train_num

            print("Train_og idx: ", self.train_og_index)
            print("Val_og idx: ", self.val_og_index)
        else:
            data_df = load_dataset(self.full_data_path)["train"]  # fixed -> for indexing all samples
            # convert to json format
            list_data_dict = []
            for i in range(len(data_df)):
                # parse data_df[i]['conversations'] from str to list
                list_data_dict.append(dict(instruction=data_df[i]['conversations'][0], output=data_df[i]['conversations'][1]))
            self.train_data = [list_data_dict[i] for i in range(len(list_data_dict))]
        
        # make a supervised data module for the valiation set
        if self.val_data is not None:
            print("Creating validation data module...")
            self.val_data = make_supervised_data_module(tokenizer=self.tokenizer, data_path=self.val_data, verbose=False)
            self.val_data["train_dataset"].ids = self.val_idx
            self.val_data["train_dataset"].train_og_index = self.val_og_index
            
        self.n_pool = len(self.train_data)
        # keep track of labeled/unlabeled (1/0) index

        self.labeled_idx = torch.zeros(self.n_pool, dtype=bool)  
        # saving options
        self.data_path_root = args["data_path_root"]
        self.output_dir_root = args["output_dir_root"]
        train_args = args["train_args"]
        train_args["output_dir"] = self.output_dir_root  # dummy init -> to update for each round
        # get the name of the transformer model
        if "t5" in self.model.__class__.__name__:
            self.training_args = Seq2SeqTrainingArguments(**train_args)
        else:
            self.training_args = TrainingArguments(**train_args)

        self.sam_mode = args.get("sam_mode", "no")
        self.sam_rho = args.get("sam_rho", 0.05)
        self.sam_adaptive = args.get("sam_adaptive", False)
        self.sam_schedule = args.get("sam_schedule", "constant")
        self.sam_schedule_warmup = args.get("sam_schedule_warmup", 0)
        self.sam_rho_min = args.get("sam_rho_min", 0)

    def initialize_labeled_data(self):
        """Randomly init labeled pool"""
        if not is_running_distributed() or torch.distributed.get_rank() == 0:
            tmp_idxs = torch.randperm(self.n_pool)  # randomly permute indices (total_data_size, )
            self.labeled_idx[tmp_idxs[:self.init_label_num]] = True  # labeled=1, unlabeled=0 (total_data_size,)

    def save_labeled_unlabeled_data(self):
        """update & save current labaled & unlabeled pool"""
        if not is_running_distributed() or torch.distributed.get_rank() == 0:
            # obtain & check labeled_idx for current round
            labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]  # self.labeled_idx -> kept upated

            labeled_data_json_format = [self.train_data[_] for _ in labeled_idx] 
            unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]
            unlabeled_data_json_format = [self.train_data[_] for _ in unlabeled_idx]
            rank0_print(f"*** labeled_idx: {labeled_idx}")
            # save current labeled & unlabeld data
            labeled_data_path = f"{self.data_path_root}/labeled.json"
            labeled_idx_path = f"{self.data_path_root}/labeled_idx.npy"
            unlabeled_data_path = f"{self.data_path_root}/unlabeled.json"
            if not is_running_distributed() or torch.distributed.get_rank() == 0:
                retry = 0
                while True:
                    jdump(labeled_data_json_format, labeled_data_path)
                    try:
                        temp_labeled = jload(labeled_data_path)
                        rank0_print(f"*** jdump(labeled_data_json_format, labeled_data_path) SUCESSFUL to --> {labeled_data_path}")
                        break
                    except:
                        retry += 1
                        rank0_print(f"*** jdump(labeled_data_json_format, labeled_data_path) FAILED to --> {labeled_data_path}")
                        if retry > 5:
                            raise
                        continue
                retry = 0
                while True:
                    jdump(unlabeled_data_json_format, unlabeled_data_path)
                    try:
                        temp_unlabeled = jload(unlabeled_data_path)
                        rank0_print(f"*** jdump(unlabeled_data_json_format, unlabeled_data_path) SUCESSFUL to --> {unlabeled_data_path}")
                        break
                    except:
                        retry += 1
                        rank0_print(f"*** jdump(unlabeled_data_json_format, unlabeled_data_path) FAILED to --> {unlabeled_data_path}")
                        if retry > 5:
                            raise
                        continue
                np.save(labeled_idx_path, labeled_idx.numpy())
    
    def get_updated_train_data(self):
        data_path = f"{self.data_path_root}/labeled.json"
        print("Getting Train Supervised Data Module with path :", data_path)
        labeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, data_path=data_path)
        return labeled_data_module
    
    def get_unlabeled_data(self):
        data_path = f"{self.data_path_root}/unlabeled.json"
        unlabeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, 
                                                                data_path=data_path)
        return unlabeled_data_module
    
    def train(self):
        print("schedule.train() called")
        data_module = self.get_updated_train_data()       
        
        #test_trainloader(data_module["train_dataset"], self.per_example_usages, self.train_og_index.cpu().tolist(), self.val_og_index.cpu().tolist())  
        #exit()

        print("Schedule base, train_og_index: ", data_module["train_dataset"].train_og_index)

        # sanity-check
        if not is_running_distributed() or torch.distributed.get_rank() == 0:
            for sanity_sample in data_module["train_dataset"]:
                break
            rank0_print(f"*** SANITY-CHECK: Training-Sample#1. - TEXT.:\n\n{self.tokenizer.decode(sanity_sample['input_ids'])}\n\n")
        
        # get validation data
        if self.val_data is not None:
            data_module["eval_dataset"] = self.val_data["train_dataset"]

        print("val_data.train_og_index: ", data_module["eval_dataset"].train_og_index)

        optimizer, lr_scheduler = self._create_optimizer_and_scheduler(
            data_module["train_dataset"]
        )

        output_dir = f"{self.output_dir_root}/"
        self.training_args.output_dir = output_dir

        if "t5" in self.model.__class__.__name__:
            trainer_cls = Seq2SeqTrainer # similarly
        elif self.sam_mode == "sam":
            trainer_cls = FSDPSAMTrainer
        elif self.sam_mode == "fsam":
            trainer_cls = FSDPFunctionalSAMTrainer
        elif self.sam_mode == "preconfsam":
            trainer_cls = FSDPFunctionalSAMTrainer
        else:
            trainer_cls = CustomTrainer

        self.training_args.remove_unused_columns = False


        if self.load_from:
            possible_checkpoints = os.listdir(self.load_from)
            if len(possible_checkpoints) == 0:
                self.load_from = None
                self.load_step = None
            else:
                # keep only folders
                if self.load_step is None:
                    possible_checkpoints = [ckpt for ckpt in possible_checkpoints if os.path.isdir(os.path.join(self.load_from, ckpt))]
                    # sort on the integer value of the final number checkpoint-x
                    possible_checkpoints = sorted(possible_checkpoints, key=lambda x: int(x.split("-")[-1]))
                    
                    # possible_checkpoints = sorted(possible_checkpoints)
                    checkpoint = possible_checkpoints[-1]
                else:
                    checkpoint = f"checkpoint-{self.load_step}"
                checkpoint_dir = os.path.join(self.load_from, checkpoint)
                print("checkpoints: ", possible_checkpoints)
                print(f"Loading from checkpoint: {checkpoint_dir}")

        #exit()
        #print("training args: ", self.training_args)
        # self.model = self.model.cpu()

        def get_auto_wrap_policy(threshold=1e6):
            def policy(module, recurse, nonwrapped_numel):
                return sum(p.numel() for p in module.parameters()) >= threshold
            return policy

        """
        if self.training_args.fsdp:
            self.model = FSDP(
                self.model.cpu(),
                auto_wrap_policy=get_auto_wrap_policy(threshold=1e6),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
                device_id=torch.cuda.current_device(),
            )
        """
        training_args = {
            'model': self.model,
            'args': self.training_args,
            'train_dataset': data_module["train_dataset"],
            'eval_dataset': data_module.get("eval_dataset", None),
            'data_collator': data_module["data_collator"],
            'tokenizer': self.tokenizer,
        }
        if not self.training_args.fsdp:
            training_args["optimizers"] = (optimizer, lr_scheduler)
        else:
            """
            auto_wrap_policy = transformer_auto_wrap_policy(self.training_args.fsdp_config)
            training_args["fsdp"] = True
            self.model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),  # optional
            )
            #"""

        # add per_examples_usages to training_args
        #training_args["per_example_usages"] = self.per_example_usages if hasattr(self, 'per_example_usages') else None


        if self.sam_mode == "no":
            trainer = trainer_cls(**training_args)
        else:
            additional_training_args = {
                'sam_mode': self.sam_mode,
                'sam_rho': self.sam_rho,
                'sam_adaptive': self.sam_adaptive,
                'schedule': self.sam_schedule,
                'schedule_warmup': self.sam_schedule_warmup,
                'rho_min': self.sam_rho_min,
            }
            training_args.update(additional_training_args)
            trainer = trainer_cls(**training_args)

        trainer.per_example_usages = self.per_example_usages if hasattr(self, 'per_example_usages') else None
        print("in schedule base, checking for per_example_usages", trainer.per_example_usages)

        rank0_print(f"*** Sampler Type: {type(trainer.get_train_dataloader().sampler)}")

        trainer.custom_load_dir = None
        if self.load_from:
            trainer.custom_load_dir = checkpoint_dir
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            trainer.train()
        
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)
        
        # check if we need ddp saving
        #if torch.distributed.get_world_size() <= 1:
        #    self.model.save_pretrained(f"{output_dir}/pretrained")
        #else:
        #    self.model.module.save_pretrained(f"{output_dir}/pretrained")
        self.model.save_pretrained(f"{output_dir}/pretrained")

    def _create_optimizer_and_scheduler(self, train_dataset):
        lr = self.training_args.learning_rate
        wd = self.training_args.weight_decay
        num_epochs = self.training_args.num_train_epochs
        batch_size = self.training_args.per_device_train_batch_size

        # Number of updates per epoch and total training steps
        num_update_steps_per_epoch = len(train_dataset) // batch_size
        max_train_steps = int(num_epochs * num_update_steps_per_epoch)

        # Base optimizer constructor
        def base_optimizer_fn(param_groups, **kwargs):
            return AdamW(param_groups, lr=lr, weight_decay=wd)

        if self.sam_mode == "sam":
            rank0_print(f"*** Using SAM: rho={self.sam_rho}, adaptive={self.sam_adaptive}")
            optimizer = SAM(
                self.model.parameters(),
                base_optimizer=base_optimizer_fn,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive,
            )

        elif self.sam_mode == "fsam":
            optimizer = FunctionalSAM(
                self.model.parameters(),
                base_optimizer=base_optimizer_fn,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive,
                precondition=False,
                schedule=self.sam_schedule,
                schedule_warmup=self.sam_schedule_warmup,
                rho_min=self.sam_rho_min,
            )
        elif self.sam_mode == "preconfsam":
            optimizer = FunctionalSAM(
                self.model.parameters(),
                base_optimizer=base_optimizer_fn,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive,
                precondition=True, 
                schedule=self.sam_schedule,
                schedule_warmup=self.sam_schedule_warmup,
                rho_min=self.sam_rho_min,
            )


        else:
            rank0_print("*** Using AdamW")
            optimizer = base_optimizer_fn(self.model.parameters())

        # Create learning rate scheduler
        print("External, max training steps: ", max_train_steps)

        #training_steps = self.training_args.num_train_epochs * len(train_dataset) // self.training_args.per_device_train_batch_size // self.training_args.gradient_accumulation_steps
        #print(f"Getting scheduler, num training steps: {training_steps}, warmup ratio: {self.training_args.warmup_ratio}")

        #lr_scheduler = get_scheduler(
        #    name=self.training_args.lr_scheduler_type,
        #    optimizer=optimizer,
        #    num_warmup_steps=int(self.training_args.warmup_ratio * max_train_steps),
        #    num_training_steps=max_train_steps,
        #)
        lr_scheduler = None

        return optimizer, lr_scheduler


from collections import Counter
from tqdm import tqdm

class OversampleWrapper(Dataset):
    def __init__(self, base_dataset, usage_counts, verbose=False):
        print("creating oversample wrapper with length: ", len(base_dataset), "and usage count sum:", sum(usage_counts))
        self.dataset = base_dataset
        self.indices = [i for i, count in enumerate(usage_counts) for _ in range(count)]
        print("Oversampled wrapper, indices: ", self.indices)

        # Optional: print sample usage distribution
        counts = Counter(self.indices)
        if verbose:
            print("[OversampleWrapper] Sample usage counts:")
            for i in sorted(counts):
                print(f"  Index {i}, OG Index {self.dataset[i]['original_idx']}: {counts[i]}x")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class DefaultLoader(Dataset):
    def __init__(self, base_dataset, verbose=False):
        self.dataset = base_dataset
        if verbose:
            print("[DefaultLoader] Initialized with dataset length:", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def test_trainloader(train_dataset, per_example_usages, expected_train_ogs, expected_val_ogs):
    print("Testing train loader...")
    print("per example usages:", per_example_usages)
    oversample_wrapper = OversampleWrapper(train_dataset, per_example_usages, verbose=True)

    seen_shuffled = []
    seen_original = []

    default_wrapper = DefaultLoader(train_dataset, verbose=True)
    default_shuffled = []
    default_original = []

    label_map = {}
    label_map_original ={}

    def _ensure_list(val):
        if isinstance(val, torch.Tensor):
            if val.ndim == 0:
                return [val.item()]
            else:
                return val.cpu().tolist()
        elif isinstance(val, list):
            return val
        else:
            return [val]


    def compare_lists(list1, list2):
        if list1 is None and list2 is None:
            return True
        if list1 is None or list2 is None:
            return False
        
        shorter_list = list1 if len(list1) < len(list2) else list2
        longer_list = list1 if len(list1) >= len(list2) else list2
        return shorter_list == longer_list[:len(shorter_list)]


    for i, batch in enumerate(default_wrapper):
        shuffled_ids = batch.get("id", None)
        original_idxs = batch.get("original_idx", None)
        print(f"Default Batch {i} shuffled IDs:", shuffled_ids, "Original indices:", original_idxs)
        for original_idx in _ensure_list(original_idxs):
            if original_idx not in label_map_original:
                label_map_original[original_idx] = batch.get("label", None)
            elif not compare_lists(label_map_original[original_idx], batch.get("label", None)):
                print(f"Warning: Original index {original_idx} has multiple labels: {label_map_original[original_idx]} and {batch.get('label', None)}")
        default_shuffled.extend(_ensure_list(shuffled_ids))
        default_original.extend(_ensure_list(original_idxs))

    for i, batch in enumerate(oversample_wrapper):
        shuffled_ids = batch.get("id", None)
        original_idxs = batch.get("original_idx", None)
        for original_idx in _ensure_list(original_idxs):
            if original_idx not in label_map:
                label_map[original_idx] = batch.get("label", None)
            elif not compare_lists(label_map[original_idx], batch.get("label", None)):
                print(f"Warning: Original index {original_idx} has multiple labels: {label_map[original_idx]} and {batch.get('label', None)}")

        print(f"Batch {i} shuffled IDs:", shuffled_ids, "Original indices:", original_idxs)
        seen_shuffled.extend(_ensure_list(shuffled_ids))
        seen_original.extend(_ensure_list(original_idxs))


    seen_shuffled = sorted(seen_shuffled)
    seen_original = sorted(seen_original)

    print("Checking seen shuffled and original indices...")
    seen_per_example_usages = np.zeros(len(train_dataset))
    for id_val in seen_shuffled:
        seen_per_example_usages[id_val] += 1

    # confirm that seen_per_examples usages matches the expected per_example_usages
    if not np.array_equal(seen_per_example_usages, per_example_usages):
        print("Mismatch in per_example_usages:")
        print("Expected:", per_example_usages)
        print("Seen:", seen_per_example_usages)
    print("Finished comparison to per_example_usages.")


    print("Seen shuffled IDs:", seen_shuffled)
    print("Seen original indices:", seen_original)
        #print(batch)

    seen_set = set(seen_original)
    expected_train_set = set(expected_train_ogs)
    expected_val_set = set(expected_val_ogs)

    # Training check — items in expected_train not in seen_original
    missing_train = expected_train_set - seen_set
    for idx in missing_train:
        print(f"Expected train original index {idx} not found in seen original indices.")

    # Validation check — items in expected_val that accidentally appear in training
    val_in_train = expected_val_set & seen_set
    for idx in val_in_train:
        print(f"Expected validation original index {idx} found in seen original indices, but it should not be in the training set.")
    exit()

    print("Test completed.")