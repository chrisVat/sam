import torch
from torch.optim import AdamW
from transformers import Trainer, get_scheduler
from tqdm.auto import tqdm
import torch.distributed as dist
from sam_functional import FunctionalSAM
#from sam_functional_preconditioned import PreconditionedFunctionalSAM
from utils import rank0_print
import torch.nn.functional as F
import gc
from transformers.trainer_pt_utils import LabelSmoother
from functorch import vjp
import time
import datetime
import os
from consts import LLAMA_IGNORE_INDEX
import numpy as np
import inspect
import contextlib
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_callback import TrainerState
from utils import load_ddp_state_dict
import numpy as np
import random
import math
from utils import is_running_distributed
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker, has_length
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler



LOW_GPU = False





if is_datasets_available():
    import datasets




class FSDPFunctionalSAMTrainer(Trainer):
    def __init__(self, *args, sam_mode="no", sam_rho=0.05, sam_adaptive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_mode = sam_mode
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.my_label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor, ignore_index=LLAMA_IGNORE_INDEX)
        self.global_step = 0 
        self.vjp_preallocated = None

        if not hasattr(self.model, "no_sync"): # not the best practices, i just want this to work quickly.
            self.model.no_sync = contextlib.nullcontext
        
        gpu_rank = dist.get_rank() if dist.is_initialized() else 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_log_rank{gpu_rank}_{kwargs['args'].run_name.replace("/", "")}_{timestamp}.txt"


    # nanfriendly version.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.autocast_smart_context_manager():
            outputs = model(**inputs)
            loss = outputs.loss
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch

        # Check for NaN on the tensor
        if torch.isnan(loss).any():
            print("NaN Loss detected.")
            loss = torch.zeros_like(loss)

        return (loss, outputs) if return_outputs else loss


    def get_minibatch_gradients(self, inputs):
        prepared_inputs = self._prepare_inputs(inputs)
        labels = prepared_inputs.get("labels")

        with self.autocast_smart_context_manager():
            outputs = self.model(**prepared_inputs, return_dict=True)
            logits = outputs.logits
            parameter_loss = self.my_label_smoother(outputs, labels, shift_labels=True)
            # move outputs to cpu
            outputs = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in outputs.items()}

        parameter_loss = parameter_loss / self.accum_steps

        logit_grad = torch.autograd.grad(
            outputs=parameter_loss,
            inputs=logits,
            retain_graph=True, 
            allow_unused=True,
        )[0].detach().cpu()

        self.accelerator.backward(parameter_loss)

        self.accumulated_logit_grads.append(logit_grad) 

        if not torch.isnan(parameter_loss):
            self.total_loss = self.total_loss + parameter_loss.item() 
            self.accumulated_pred_loss = self.accumulated_pred_loss + parameter_loss.item()

        prepared_inputs = {
            key: (value.cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in prepared_inputs.items()
        }

        self.accumulated_inputs.append(prepared_inputs)
        return 


    def sync_grads(self):
        for p in self.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= dist.get_world_size()


    def load_checkpoint(self, checkpoint_path):
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu")) #, map_location=self.model.device))
        self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu")) #, map_location=self.model.device))
        self.model.load_state_dict(load_ddp_state_dict(checkpoint_path))
        checkpoint_state_file = os.path.join(checkpoint_path, "trainer_state.json")
        self.state = TrainerState.load_from_json(checkpoint_state_file)
        self._load_rng_state(checkpoint_path)



    def _inner_training_loop(self, *args, **kwargs):
        #eval_results = self.evaluate()
        #self.log(eval_results)
        #rank0_print(f"Initial evaluation results: {eval_results}")
        #rank0_print(f"Pre Load GPU Usage: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        
        # load first so there's no random orderings with loading checkpoints
        train_dataloader = self.get_train_dataloader()

        
        print("Train Dataloader", train_dataloader)
        print("train loader length: ", len(train_dataloader))
    

        with torch.no_grad():
            if self.custom_load_dir is not None:
                self.load_checkpoint(self.custom_load_dir)

        self.model.train() 


        accum_steps = (
            self.args.gradient_accumulation_steps
            if self.args.gradient_accumulation_steps > 0
            else 1
        )
        self.accum_steps = accum_steps

        total_batches = len(train_dataloader)
        total_updates_per_epoch = (total_batches + accum_steps - 1) // accum_steps

        progress_bar = tqdm(
            total=total_updates_per_epoch * int(self.args.num_train_epochs),
            desc=f"Rank {dist.get_rank() if is_running_distributed() else 0} Training",
        )

        self.total_loss = 0.0
        global_step = getattr(self.state, "global_step", 0)

        cur_steps = -1

        if self.custom_load_dir is not None:
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            self.model.zero_grad()

        #rank0_print(f"Starting GPU Usage: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

        for epoch in range(int(self.args.num_train_epochs)):
            self.cur_epoch = epoch

            self.epoch_loss = 0.0
            num_batches = 0
            updates_this_epoch = 0
            self.accumulated_pred_loss = 0.0

            self.accumulated_inputs = []  # List to store prepared inputs.
            self.accumulated_logit_grads = []

            for inputs in train_dataloader:                
                cur_steps += 1
                # skip ahead if we are loading from a checkpoint
                if cur_steps < global_step*accum_steps: 
                    if cur_steps % accum_steps == 0 and cur_steps > 0:
                        progress_bar.update(1)
                    del inputs
                    continue


                #rank0_print(f"Getting Minibatch - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                with self.model.no_sync():
                    self.get_minibatch_gradients(inputs)
                #rank0_print(f"Post Computing Minibatch Gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                
                if len(self.accumulated_inputs) == accum_steps:
                    if is_running_distributed():
                        self.sync_grads()
                    with self.model.no_sync():
                        #rank0_print(f"Pre Perturbation - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        
                        if self.optimizer.precondition and LOW_GPU:
                            self.optimizer.move_adamw_second_moment_to_gpu(second_only=True)
                        
                        self.optimizer.first_step_functional(zero_grad=True) # , warmup=global_step<=MIN_WARMUP_STEPS)
                        
                        #if self.optimizer.precondition:
                        #    self.optimizer.move_adamw_second_moment_to_cpu()
                            
                        #rank0_print(f"Post Perturbation - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        # moves old params to cpu, optional depending on gpu usage
                        self.optimizer.move_old_to_cpu()
                        #self.optimizer.move_optimizer_to_cpu()
                        #rank0_print(f"Calling Second Step Functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        self.second_step_functional() 
                        #rank0_print(f"Done Second Step Functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    if is_running_distributed():
                        self.sync_grads()

                    #rank0_print(f"Moving Optimizer to GPU: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    #self.optimizer.move_optimizer_to_gpu()
                    #rank0_print(f"restored GPU parameters - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    self.optimizer.restore_old()

                    #rank0_print(f"Moving Old to GPU - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    #self.optimizer.move_old_to_gpu()
                    #rank0_print(f"Post Moments to GPU - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    
                    if LOW_GPU:
                        self.optimizer.move_adamw_second_moment_to_gpu()

                    cur_grad_norm = self.optimizer._grad_norm().item()

                    #rank0_print(f"Final Step - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    self.optimizer.final_step(zero_grad=True, restore_old=False)
                    #rank0_print(f"Post Final Step - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    if LOW_GPU:
                        self.optimizer.move_adamw_second_moment_to_cpu()
                    #rank0_print(f"Post Moving AdamW Second Moment to CPU - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    #self.optimizer.inspect_optimizer_state()

                    #self.optimizer.move_adamw_second_moment_to_cpu()
                    #self.optimizer.move_old_to_cpu()

                    num_batches += 1

                    #gc.collect()
                    #torch.cuda.empty_cache()

                    #rank0_print(f"Batch End - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    # --- Reset accumulators ---
                    self.accumulated_inputs = []
                    self.accumulated_logit_grads = []

                    # only step for rank 0
                    if self.lr_scheduler is not None and (not is_running_distributed() or dist.get_rank() == 0):
                        self.lr_scheduler.step()
                    global_step += 1
                    updates_this_epoch += 1
                    epoch_float = epoch + ((global_step % total_updates_per_epoch) / total_updates_per_epoch)

                    self.epoch_loss += self.accumulated_pred_loss

                    logs = {
                        "loss": round(self.accumulated_pred_loss, 4),
                        "avg_epoch_loss": round((self.epoch_loss / num_batches), 4),
                        "learning_rate": self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else None,
                        "epoch": round(epoch_float, 2),
                        "grad_norm": round(cur_grad_norm, 9),
                    }
                    self.state.global_step = global_step
                    self.callback_handler.on_log(self.args, self.state, self.control, logs)
                    #logs["rank"] = dist.get_rank()
                    print(logs)

                    if self.state.global_step % self.args.save_steps == 0:
                        if not is_running_distributed() or dist.get_rank() == 0:
                            print("Saving model checkpoint at global step: ", self.state.global_step)
                            with self.model.no_sync():
                                self._save_checkpoint(self.model, trial=None)
                        else:
                            self.store_flos()
                    self.accumulated_pred_loss = 0.0  # Reset logging accumulator.
                    progress_bar.update(1)
                    progress_bar.set_postfix(logs)

            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                print(f"Epoch {epoch+1} finished. Awaiting evaluation...")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    self.model.zero_grad(set_to_none=True)  # Free gradient memory
                    torch.cuda.synchronize()
                
                self.can_return_loss = True
                rank0_print("*** Beginning Evaluation ***")
                with torch.no_grad(): 
                    self.model.eval()
                    eval_results = self.evaluate()
                    self.model.train()
                
                rank0_print(f"Epoch {epoch+1} evaluation results: {eval_results}")
                
                eval_results = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in eval_results.items()}
                self.log(eval_results)
                #del eval_results

                self.model.zero_grad(set_to_none=True)  # Free gradient memory
                torch.cuda.synchronize()
                #gc.collect()
                time.sleep(5)
            
            avg_epoch_loss = self.epoch_loss / num_batches if num_batches > 0 else 0
            rank0_print(f"Epoch {epoch+1} finished. Average training loss: {avg_epoch_loss:.4f}")

        progress_bar.close()
        return self.total_loss


    def create_optimizer(self):
        if self.optimizer is None:
            lr = self.args.learning_rate
            wd = self.args.weight_decay

            def base_optimizer_fn(param_groups):
                return AdamW(param_groups, lr=lr, weight_decay=wd)

            if self.sam_mode == "fsam":
                self.optimizer = FunctionalSAM(
                    self.model.parameters(),
                    base_optimizer=base_optimizer_fn,
                    rho=self.sam_rho,
                    adaptive=self.sam_adaptive,
                    precondition=False
                )
            elif self.sam_mode == "preconfsam":
                self.optimizer = FunctionalSAM(
                    self.model.parameters(),
                    base_optimizer=base_optimizer_fn,
                    rho=self.sam_rho,
                    adaptive=self.sam_adaptive,
                    precondition=True
                )
            else:
                self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)


    def create_scheduler(self, num_training_steps: int):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=int(self.args.warmup_ratio * num_training_steps),
                num_training_steps=num_training_steps,
            )

    @torch.no_grad()
    def second_step_functional(self):
        #rank0_print(f"Second step functional, initial - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        
        with torch.no_grad():
            # Determine the maximum sequence length among the accumulated logit gradients.
            max_seq_len = max(grad.shape[1] for grad in self.accumulated_logit_grads)

            def network_fn(params, batch):
                return torch.func.functional_call(self.model, params, (), batch).logits

            perturbed_params = {name: param for name, param in self.model.named_parameters()}

            def compute_vjp_grads(microbatch, cur_avg_logit_grad):
                vjp_fn = torch.func.vjp(lambda theta: network_fn(theta, microbatch), perturbed_params)[1]
                return vjp_fn(cur_avg_logit_grad)[0]
            
            def accumulate_gradients(vjp_gradient, microbatch_count):
                model_params = dict(self.model.named_parameters())
                for name, grad in vjp_gradient.items():
                    if grad is not None:
                        param = model_params.get(name)
                        if param is not None:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param, device=self.model.device)
                            param.grad.add_(grad) # .div_(microbatch_count)
                        #grad.zero_()

            num_microbatches = 0
            # Process each accumulated input and corresponding logit gradient.
            for batch_cpu, cur_logit_grad in zip(self.accumulated_inputs, self.accumulated_logit_grads):
                batch_size = batch_cpu["input_ids"].shape[0]
                # Determine microbatch size based on max_seq_len.

                #next_best = batch_size // 2
                #next_best = 1
                #microbatch_size = batch_size if max_seq_len <= 200 else max(next_best, 1)
                microbatch_size = 1
                microbatch_count = (batch_size + microbatch_size - 1) // microbatch_size

                # Process the batch in microbatches.
                for i in range(0, batch_size, microbatch_size):
                    # Slice the microbatch from the CPU batch and move it to GPU.
                    microbatch = {
                        k: v[i:i+microbatch_size].detach().to(self.model.device)
                        for k, v in batch_cpu.items()
                    }
                    seq_len = microbatch["input_ids"].shape[1]
                    # Select and move the corresponding slice of the logit gradients.
                    cur_avg_logit_grad = (
                        cur_logit_grad[i:i+microbatch_size, :seq_len]
                        .contiguous().detach().to(self.model.device)
                    )
                    accumulate_gradients(compute_vjp_grads(microbatch, cur_avg_logit_grad), microbatch_count)
                    num_microbatches += 1
                    
                    #rank0_print(f"After accumulate gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    # clear memory
                    for k, v in microbatch.items():
                        del v
                    del microbatch
                    del cur_avg_logit_grad
                del batch_cpu, cur_logit_grad

            # scale the accumulated gradients
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.div_(num_microbatches)


            self.accumulated_inputs = []
            self.accumulated_logit_grads = []

    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Ensure that a shared generator exists.
        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(42)
        generator = self._generator


        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator
            )

        else:
            return RandomSampler(self.train_dataset, generator=generator)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))




def model_grad_l2_norm(model):
    total_norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2) if p.grad is not None else torch.tensor(0.0, device=p.device)
            for p in model.parameters()
        ]), p=2)
    return total_norm.item()




def debug_gpu_variables(var_dict, prefix=""):
    """
    Recursively checks variables in var_dict (from locals()) and prints only those using CUDA memory.
    
    Args:
        var_dict (dict): Typically `locals()` from the current function.
        prefix (str): Optional prefix for variable names.
    """
    for var_name, var_value in var_dict.items():
        full_name = f"{prefix}.{var_name}" if prefix else var_name

        # Check if it's a Tensor on CUDA
        if isinstance(var_value, torch.Tensor) and var_value.is_cuda:
            rank0_print(f"[GPU] {full_name}: {var_value.shape}, dtype={var_value.dtype}, device={var_value.device}")

        # Check if it's a list or tuple (recursively check elements)
        elif isinstance(var_value, (list, tuple)):
            for i, item in enumerate(var_value):
                if isinstance(item, torch.Tensor) and item.is_cuda:
                    rank0_print(f"[GPU] {full_name}[{i}]: {item.shape}, dtype={item.dtype}, device={item.device}")

        # Check if it's a dictionary (recursively check values)
        elif isinstance(var_value, dict):
            for key, item in var_value.items():
                if isinstance(item, torch.Tensor) and item.is_cuda:
                    rank0_print(f"[GPU] {full_name}[{key}]: {item.shape}, dtype={item.dtype}, device={item.device}")

        # Check if it's an object with attributes that might have CUDA tensors
        elif hasattr(var_value, "__dict__"):
            for attr_name, attr_value in vars(var_value).items():
                if isinstance(attr_value, torch.Tensor) and attr_value.is_cuda:
                    rank0_print(f"[GPU] {full_name}.{attr_name}: {attr_value.shape}, dtype={attr_value.dtype}, device={attr_value.device}")

def find_persistent_cuda_tensors():
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                rank0_print(f"Persistent CUDA Tensor: {obj.shape}, dtype={obj.dtype}, device={obj.device}")
        except:
            pass  # Ignore objects that cause errors

