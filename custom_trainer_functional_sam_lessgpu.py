import torch
from torch.optim import AdamW
from transformers import Trainer, get_scheduler
from tqdm.auto import tqdm
import torch.distributed as dist
from sam_functional import FunctionalSAM
from sam_functional_preconditioned import PreconditionedFunctionalSAM
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

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

import contextlib


LOG_PRED_LOSS = False
LOG_FOLDER = "loss_logs/"

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)


def _is_peft_model(model):
    return False

class FSDPFunctionalSAMTrainer(Trainer):
    def __init__(self, *args, sam_mode="no", sam_rho=0.05, sam_adaptive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_mode = sam_mode
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.my_label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor, ignore_index=LLAMA_IGNORE_INDEX)
        self.global_step = 0 
        self.vjp_preallocated = None

        #print(kwargs)
        #print(type(kwargs))
        #print(kwargs['args'])
        #print(kwargs['args'].run_name)
        #exit()

        if not hasattr(self.model, "no_sync"): # not the best practices, i just want this to work quickly.
            self.model.no_sync = contextlib.nullcontext

        gpu_rank = dist.get_rank() if dist.is_initialized() else 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_log_rank{gpu_rank}_{kwargs['args'].run_name.replace("/", "")}_{timestamp}.txt"
        self.log_file_path = os.path.join(LOG_FOLDER, log_filename)
        
        # non ddp
        #self.model.gradient_checkpointing_enable()
        # ddp
        #self.model.module.gradient_checkpointing_enable()


    def get_param_loss(self, logits, labels):
        # Shift logits and labels so that each prediction corresponds to the next token.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1)
        )
        per_token_loss = per_token_loss.view(shifted_labels.shape)
        
        valid_mask = shifted_labels != -100
        return per_token_loss.sum() / valid_mask.sum()

    def get_param_loss_v2(self, logits, labels):
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1)
        )
        return loss


    def get_minibatch_gradients(self, inputs):
        #rank0_print(f"Getting minibatch - Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        prepared_inputs = self._prepare_inputs(inputs)
        #self.accumulated_inputs.append(prepared_inputs)  # Save for later perturb pass
        parameter_loss_individual = []

        labels = prepared_inputs.get("labels")

        with self.autocast_smart_context_manager():
            outputs = self.model(**prepared_inputs, return_dict=True)
            parameter_loss = self.my_label_smoother(outputs, labels, shift_labels=True)
            outputs = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in outputs.items()}

        parameter_loss = parameter_loss / self.accum_steps

        self.accelerator.backward(parameter_loss)
        self.total_loss = self.total_loss + parameter_loss.item() 
        self.accumulated_pred_loss = self.accumulated_pred_loss + parameter_loss.item()

        del outputs, parameter_loss
        
        with self.autocast_smart_context_manager():
            outputs = self.model(**prepared_inputs, return_dict=True)
            logits = outputs.logits
            parameter_loss = self.my_label_smoother(outputs, labels, shift_labels=True)
            outputs = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in outputs.items()}

        parameter_loss = parameter_loss / self.accum_steps

        logit_grad = torch.autograd.grad(
            outputs=parameter_loss,
            inputs=logits,
            retain_graph=False, 
            allow_unused=True,
        )[0].detach().cpu()


        self.accumulated_logit_grads.append(logit_grad) 

        if LOG_PRED_LOSS:        
            sample_ids = inputs["id"].tolist()
            log_entries = "\n".join(f"{self.cur_epoch},{sample_id},{loss.item()}" for sample_id, loss in zip(sample_ids, parameter_loss_individual)) + "\n"

            with open(self.log_file_path, "a") as log_file:
                log_file.write(log_entries)

        #rank0_print(f"Logged {len(sample_ids)} samples to {log_filename}")


        prepared_inputs = {
            key: (value.cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in prepared_inputs.items()
        }

        self.accumulated_inputs.append(prepared_inputs)
        #rank0_print(f"Data sequence lengths: {self.accumulated_inputs[0]['input_ids'].shape[1]} ")
        # move labels to cpu
        #labels = labels.cpu()
        #logits = logits.cpu()


        #del outputs, logits, parameter_loss, logit_grad, prepared_inputs, labels, parameter_loss_individual
        #rank0_print(f"After deletion - Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
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

        # MIN_WARMUP_STEPS = 1000


        #self.args.num_train_epochs = 10

        for epoch in range(int(self.args.num_train_epochs)):
            self.cur_epoch = epoch
            #rank0_print(f"Epoch: {epoch}")
            #rank0_print(f"model training: {self.model.training}")

            self.epoch_loss = 0.0
            num_batches = 0
            updates_this_epoch = 0
            self.accumulated_pred_loss = 0.0

            self.accumulated_inputs = []  # List to store prepared inputs.
            self.accumulated_logit_grads = []

            for inputs in train_dataloader:                
                #print("Batch from dataloader keys: ", inputs.keys())
                
                #rank0_print(f"Num Batches: {num_batches}")
                #if num_batches > 50: # testing memory after eval
                #    break
                
                rank0_print(f"Getting Minibatch - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                
                with self.model.no_sync():
                    self.get_minibatch_gradients(inputs)
                rank0_print(f"Post Computing Minibatch Gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
            
                if len(self.accumulated_inputs) == accum_steps:
                    

                    with self.model.no_sync():
                        #rank0_print(f"Data sequence lengths: {self.accumulated_inputs[0]['input_ids'].shape[1]} ")
                        #rank0_print(f"Grad norm after minibatch accumulation: {model_grad_l2_norm(self.model)}")
                        self.optimizer.move_adamw_second_moment_to_cpu()
                        rank0_print(f"Pre Perturbation - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        self.optimizer.first_step_functional(zero_grad=True) # , warmup=global_step<=MIN_WARMUP_STEPS)
                        #rank0_print(f"First Step Function - GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        rank0_print(f"Post Perturbation - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        #rank0_print(f"Grad norm after first step model: {model_grad_l2_norm(self.model)}")
                        self.optimizer.move_old_to_cpu()
                        self.optimizer.move_optimizer_to_cpu()
                        rank0_print(f"Moved Old Model to CPU - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        #time.sleep(0.5)
                        rank0_print(f"Calling Second Step Functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        self.second_step_functional() 
                        rank0_print(f"Post Second Step Functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                        #self.optimizer.move_old_to_gpu()             
                        time.sleep(0.5)
                        #rank0_print(f"Grad norm after second step: {model_grad_l2_norm(self.model)}")
                        
                        #rank0_print(f"Second Step Functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    
                    for p in self.model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= dist.get_world_size()

                    rank0_print(f"Post All Reduce - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    # print(f"Rank {dist.get_rank()} total grad norm: {model_grad_l2_norm(self.model)}")
                    self.optimizer.move_adamw_second_moment_to_gpu()
                    self.optimizer.move_optimizer_to_gpu()
                    rank0_print(f"Post Move Optimizer to GPU - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    self.optimizer.final_step(combined=True)
                    rank0_print(f"AFterFinal Step - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    rank0_print(f"After Zeroing gradients GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    #rank0_print(f"After Zeroing gradients GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    num_batches += 1

                    #gc.collect()
                    #torch.cuda.empty_cache()

                    #rank0_print(f"Batch End - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    # --- Reset accumulators ---
                    self.accumulated_inputs = []
                    self.accumulated_logit_grads = []

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
                    self.callback_handler.on_log(self.args, self.state, self.control, logs)
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

            if self.sam_mode == "prefsam":
                self.optimizer = FunctionalSAM(
                    self.model.parameters(),
                    base_optimizer=base_optimizer_fn,
                    rho=self.sam_rho,
                    adaptive=self.sam_adaptive,
                )
            elif self.sam.mode == "prefuncsam":
                self.optimizer = PreconditionedFunctionalSAM(
                    self.model.parameters(),
                    base_optimizer=base_optimizer_fn,
                    rho=self.sam_rho,
                    adaptive=self.sam_adaptive,
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
                            param.grad.add_(grad).div_(microbatch_count)
                        grad.zero_()


            # Process each accumulated input and corresponding logit gradient.
            for batch_cpu, cur_logit_grad in zip(self.accumulated_inputs, self.accumulated_logit_grads):
                batch_size = batch_cpu["input_ids"].shape[0]
                # Determine microbatch size based on max_seq_len.

                next_best = batch_size // 2
                next_best = 1
                microbatch_size = batch_size if max_seq_len <= 200 else max(next_best, 1)
                microbatch_size = 1
                #microbatch_size = 1
                microbatch_count = (batch_size + microbatch_size - 1) // microbatch_size

                # Process the batch in microbatches.
                for i in range(0, batch_size, microbatch_size):
                    #print(f"GPU {dist.get_rank()} Minibatch: ", i)
                    rank0_print(f"\tSecond Step Gettign Microbatch: {i} - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    # Slice the microbatch from the CPU batch and move it to GPU.
                    microbatch = {
                        k: v[i:i+microbatch_size].detach().to(self.model.device)
                        for k, v in batch_cpu.items()
                    }
                    rank0_print(f"\tSecond Step After Moving Microbatch to GPU: {i} - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    seq_len = microbatch["input_ids"].shape[1]
                    # Select and move the corresponding slice of the logit gradients.
                    cur_avg_logit_grad = (
                        cur_logit_grad[i:i+microbatch_size, :seq_len]
                        .contiguous().detach().to(self.model.device)
                    )
                    rank0_print(f"\tSecond Step After Moving Logit Grad to GPU: {i} - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    # Compute the vjp gradients for this microbatch.
                    #vjp_grads = compute_vjp_grads(microbatch, cur_avg_logit_grad)
                    #rank0_print(f"Before compute vjp grads - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    #compute_vjp_grads(microbatch, cur_avg_logit_grad)
                    #rank0_print(f"After compute vjp grads - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    # Accumulate gradients in-place.
                    accumulate_gradients(compute_vjp_grads(microbatch, cur_avg_logit_grad), microbatch_count)
                    rank0_print(f"\tSecond Step After Accumulate Gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    #rank0_print(f"After accumulate gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")

                    # move back to cpu
                    for k, v in microbatch.items():
                    #    microbatch[k] = v.cpu()
                        del v
                    del microbatch
                    #cur_avg_logit_grad = cur_avg_logit_grad.cpu()
                    del cur_avg_logit_grad
                    rank0_print(f"\tSecond Step After Moving Microbatch to CPU: {i} - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                del batch_cpu, cur_logit_grad

            # move everything to cpu
            #for name, param in perturbed_params.items():
            #    perturbed_params[name] = param.cpu()
            
            #for name, grad in self.vjp_preallocated.items():
            #    self.vjp_preallocated[name] = grad.cpu()

            # Reset accumulators so that the model now has the updated gradients.
            self.accumulated_inputs = []
            self.accumulated_logit_grads = []

        #debug_gpu_variables(locals(), prefix="SecondStepFunctional")
        #rank0_print(f"Second step functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")




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

