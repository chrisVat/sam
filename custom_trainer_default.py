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

LOG_PRED_LOSS = True
LOG_FOLDER = "loss_logs/"

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)


 # 2,3 for retraining for 05

class DefaultTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        self.global_step = 0 
        self.vjp_preallocated = None

        #print(kwargs)
        #print(type(kwargs))
        #print(kwargs['args'])
        #print(kwargs['args'].run_name)
        #exit()

        gpu_rank = dist.get_rank() if dist.is_initialized() else 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_log_rank{gpu_rank}_{kwargs['args'].run_name.replace("/", "")}_{timestamp}.txt"
        self.log_file_path = os.path.join(LOG_FOLDER, log_filename)


    def get_minibatch_gradients(self, inputs):
        #rank0_print(f"Getting minibatch - Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        prepared_inputs = self._prepare_inputs(inputs)
        #self.accumulated_inputs.append(prepared_inputs)  # Save for later perturb pass

        labels = prepared_inputs.get("labels")

        with self.autocast_smart_context_manager():
            outputs = self.model(**prepared_inputs)
            logits = outputs.logits
            parameter_loss = self.label_smoother(outputs, labels, shift_labels=True)

        parameter_loss = parameter_loss / self.accum_steps

        self.accelerator.backward(parameter_loss)
        return 

    # core training loop for functional sam
    def _inner_training_loop(self, *args, **kwargs):
        #print("!!!!!!!\n\n\n!!!!!!!!\n\n\n!!!!! Inner training loop!!!!!!!")
        #eval_results = self.evaluate()
        #self.log(eval_results)
        #rank0_print(f"Initial evaluation results: {eval_results}")

        self.model.train() 
        train_dataloader = self.get_train_dataloader()
        print("Train Dataloader", train_dataloader)

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
            desc=f"Rank {dist.get_rank()} Training",
        )
        self.total_loss = 0.0
        global_step = 0

        for epoch in range(int(self.args.num_train_epochs)):
            self.cur_epoch = epoch
            #rank0_print(f"Epoch: {epoch}")
            #rank0_print(f"model training: {self.model.training}")

            self.epoch_loss = 0.0
            num_batches = 0
            updates_this_epoch = 0
            self.accumulated_pred_loss = 0.0


            for inputs in train_dataloader:                
                self.get_minibatch_gradients(inputs)
                #rank0_print(f"Get Minibatch Gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
            
                if len(self.accumulated_inputs) == accum_steps:
                    """
                    for p in self.model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= dist.get_world_size()
                    """
                    total_grad_sum = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_grad_sum += p.grad.sum()

                    print(f"Rank {dist.get_rank()} total grad sum: {total_grad_sum.item()}")

                    
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    num_batches += 1

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    global_step += 1
                    updates_this_epoch += 1
                    epoch_float = epoch + (updates_this_epoch / total_updates_per_epoch)

                    self.epoch_loss += self.accumulated_pred_loss

                    logs = {
                        "pred_loss": round(self.accumulated_pred_loss, 4),
                        "avg_epoch_loss": round((self.epoch_loss / num_batches), 4),
                        "learning_rate": round(self.lr_scheduler.get_last_lr()[0], 6) if self.lr_scheduler else None,
                        "epoch": round(epoch_float, 2),
                    }
                    self.state.global_step = global_step

                    logs["rank"] = dist.get_rank()
                    print(logs)

                    if self.state.global_step % self.args.save_steps == 0:
                        if dist.get_rank() == 0:
                            print("Saving model checkpoint at global step: ", self.state.global_step)
                            with self.model.no_sync():
                                self._save_checkpoint(self.model, trial=None)
                        else:
                            self.store_flos()

                    self.accumulated_pred_loss = 0.0  # Reset logging accumulator.
                    progress_bar.update(1)
                    progress_bar.set_postfix(logs)

            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                print(f"Epoch {epoch+1} finished on rank {dist.get_rank()}. Awaiting evaluation...")
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

