import numpy as np
import json
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler, AdamW
from utils import jload, jdump, make_supervised_data_module, get_model, rank0_print
from sam import SAM
#from functional_sam import PreconditionedFunctionalSAM
from custom_trainer_sam import FSDPSAMTrainer
from custom_trainer_functional_sam_memtest import FSDPFunctionalSAMTrainer
from sam_functional import FunctionalSAM 
from sam_functional_preconditioned import PreconditionedFunctionalSAM
# ddp
from torch.nn.parallel import DistributedDataParallel as DDP


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
            #dataset_shuffled = raw_dataset.shuffle(seed=42)
            #train_num = int(len(dataset_shuffled) * 0.95)
            #train_data = dataset_shuffled.select(range(train_num))
            #val_data = dataset_shuffled.select(range(train_num, len(dataset_shuffled)))
            split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)
            train_data = split_dataset["train"]
            val_data = split_dataset["test"]
            # use keep only for train data and val data
            #keep_only, keep_only_train = 500, 250
            #train_data = train_data.select(range(keep_only_train))
            #val_data = val_data.select(range(keep_only))
            
            self.train_data = [train_data[i] for i in range(len(train_data))]
            self.val_data = [val_data[i] for i in range(len(val_data))]
            self.train_idx = torch.arange(len(self.train_data))
            self.val_idx = torch.arange(len(self.val_data))
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
            self.val_data = make_supervised_data_module(tokenizer=self.tokenizer, data_path=self.val_data)
            
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

    def initialize_labeled_data(self):
        """Randomly init labeled pool"""
        if torch.distributed.get_rank() == 0:
            tmp_idxs = torch.randperm(self.n_pool)  # randomly permute indices (total_data_size, )
            self.labeled_idx[tmp_idxs[:self.init_label_num]] = True  # labeled=1, unlabeled=0 (total_data_size,)

    def save_labeled_unlabeled_data(self):
        """update & save current labaled & unlabeled pool"""
        if torch.distributed.get_rank() == 0:
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
            if torch.distributed.get_rank() == 0:
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
        labeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, data_path=data_path)
        return labeled_data_module
    
    def get_unlabeled_data(self):
        data_path = f"{self.data_path_root}/unlabeled.json"
        unlabeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, 
                                                                data_path=data_path)
        return unlabeled_data_module
    
    def train(self):
        data_module = self.get_updated_train_data()
        # sanity-check
        if torch.distributed.get_rank() == 0:
            for sanity_sample in data_module["train_dataset"]:
                break
            rank0_print(f"*** SANITY-CHECK: Training-Sample#1. - TEXT.:\n\n{self.tokenizer.decode(sanity_sample['input_ids'])}\n\n")
        
        # get validation data
        if self.val_data is not None:
            data_module["eval_dataset"] = self.val_data["train_dataset"]

        optimizer, lr_scheduler = self._create_optimizer_and_scheduler(
            data_module["train_dataset"]
        )

        output_dir = f"{self.output_dir_root}/"
        self.training_args.output_dir = output_dir

        if "t5" in self.model.__class__.__name__:
            trainer_cls = Seq2SeqTrainer # similarly
        elif self.sam_mode == "sam":
            trainer_cls = FSDPSAMTrainer
        elif self.sam_mode == "prefsam":
            trainer_cls = FSDPFunctionalSAMTrainer
        elif self.sam_mode == "prefuncsam":
            trainer_cls = FSDPFunctionalSAMTrainer
        else:
            trainer_cls = Trainer

        """
        trainer_cls = Seq2SeqTrainer if "t5" in self.model.__class__.__name__ else Trainer
        trainer = trainer_cls(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            **data_module,
            optimizers=(optimizer, lr_scheduler),
        )
        """

        # print model layers, (get fc layer name)
        #print(self.model)
        #exit()

        self.training_args.remove_unused_columns = False

        local_rank = torch.distributed.get_rank()
        self.model.to(local_rank)
        if torch.distributed.get_world_size() > 1:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)


        if self.sam_mode == "no":
            trainer = trainer_cls(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                **data_module,
                optimizers=(optimizer, lr_scheduler),
            )
        else:
            trainer = trainer_cls(
                model=self.model,
                args=self.training_args,
                train_dataset=data_module["train_dataset"],
                eval_dataset=data_module.get("eval_dataset", None),
                data_collator=data_module["data_collator"],
                tokenizer=self.tokenizer,
                optimizers=(optimizer, lr_scheduler),
                sam_mode=self.sam_mode,
                sam_rho=self.sam_rho,
                sam_adaptive=self.sam_adaptive,
            )

        rank0_print(f"*** Sampler Type: {type(trainer.get_train_dataloader().sampler)}")


        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)
        #self.model.save_pretrained(f"{output_dir}/pretrained")
        self.model.module.save_pretrained(f"{output_dir}/pretrained")

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

        elif self.sam_mode == "prefsam":
            rank0_print(f"*** Using Functional SAM: rho={self.sam_rho}, adaptive={self.sam_adaptive}")

            optimizer = FunctionalSAM(
                self.model.parameters(),
                base_optimizer=base_optimizer_fn,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive)

        elif self.sam_mode == "prefuncsam":
            rank0_print(f"*** Using Preconditioned Functional SAM: rho={self.sam_rho}, adaptive={self.sam_adaptive}")
            optimizer = PreconditionedFunctionalSAM(
                self.model.parameters(),
                base_optimizer=base_optimizer_fn,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive
            )


        else:
            rank0_print("*** Using AdamW")
            optimizer = base_optimizer_fn(self.model.parameters())

        # Create learning rate scheduler
        lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(self.training_args.warmup_ratio * max_train_steps),
            num_training_steps=max_train_steps,
        )

        return optimizer, lr_scheduler
