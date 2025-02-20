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


 # 2,3 for retraining for 05

class FSDPFunctionalSAMTrainer(Trainer):
    def __init__(self, *args, sam_mode="no", sam_rho=0.05, sam_adaptive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_mode = sam_mode
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        self.global_step = 0 

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

        logit_grad = torch.autograd.grad(
            outputs=parameter_loss,
            inputs=logits,
            retain_graph=True, 
            allow_unused=True,
        )[0]

        self.accelerator.backward(parameter_loss)

        logit_grad = logit_grad.detach() 

        self.accumulated_logit_grads.append(logit_grad.detach().cpu()) 

        self.total_loss = self.total_loss + parameter_loss.item() 
        self.accumulated_pred_loss = self.accumulated_pred_loss + parameter_loss.item()
        prepared_inputs_cpu = {}
        for key, value in prepared_inputs.items():
            if isinstance(value, torch.Tensor):
                prepared_inputs_cpu[key] = value.cpu()
            else:
                prepared_inputs_cpu[key] = value

        self.accumulated_inputs.append(prepared_inputs_cpu)


        del outputs, logits, parameter_loss, logit_grad, prepared_inputs, labels
        #rank0_print(f"After deletion - Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        return 

    # core training loop for functional sam
    def _inner_training_loop(self, *args, **kwargs):
        #eval_results = self.evaluate()
        #self.log(eval_results)
        #rank0_print(f"Initial evaluation results: {eval_results}")

        self.model.train() 
        train_dataloader = self.get_train_dataloader()
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

        MIN_WARMUP_STEPS = 1000


        #self.args.num_train_epochs = 10

        for epoch in range(int(self.args.num_train_epochs)):
            #rank0_print(f"Epoch: {epoch}")
            #rank0_print(f"model training: {self.model.training}")

            self.epoch_loss = 0.0
            num_batches = 0
            updates_this_epoch = 0
            self.accumulated_pred_loss = 0.0

            self.accumulated_inputs = []  # List to store prepared inputs.
            self.accumulated_logit_grads = []

            for inputs in train_dataloader:                
                #rank0_print(f"Num Batches: {num_batches}")
                if num_batches > 50: # testing memory after eval
                    break
                
                #rank0_print(f"Load Inputs - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                self.get_minibatch_gradients(inputs)
                #rank0_print(f"Get Minibatch Gradients - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
            
                if len(self.accumulated_inputs) == accum_steps:
                    #rank0_print(f"Grad norm after minibatch accumulation: {model_grad_l2_norm(self.model)}")
                    self.optimizer.first_step_functional(zero_grad=True, warmup=global_step<=MIN_WARMUP_STEPS)
                    #rank0_print(f"First Step Function - GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    #rank0_print(f"Perturbed Model - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    #rank0_print(f"Grad norm after first step model: {model_grad_l2_norm(self.model)}")
                    with torch.no_grad():
                        self.second_step_functional()              
                    #rank0_print(f"Grad norm after second step: {model_grad_l2_norm(self.model)}")
                    
                    #rank0_print(f"Second Step Functional - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
                    self.optimizer.final_step(zero_grad=True)

                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    num_batches += 1

                    del inputs
                    for accum_dict in self.accumulated_inputs:
                        for k, v in accum_dict.items():
                            v.detach()#.cpu()
                            del k, v
                        del accum_dict
                    
                    del self.accumulated_inputs[:]
                    del self.accumulated_logit_grads[:]
                    
                    for grad in self.accumulated_logit_grads:
                        grad.detach() 
                        del grad

                    gc.collect()
                    torch.cuda.empty_cache()

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

                    self.accumulated_pred_loss = 0.0  # Reset logging accumulator.
                    progress_bar.update(1)
                    progress_bar.set_postfix(logs)

            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                print(f"Epoch {epoch+1} finished on rank {dist.get_rank()}. Awaiting evaluation...")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    
                    # clear out memory
                    self.model.zero_grad(set_to_none=True)  # Free gradient memory
                    torch.cuda.synchronize()
                    gc.collect()

                    #torch.cuda.empty_cache()
                    #rank0_print(f"Pre Eval - GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                
                
                rank0_print("*** Beginning Evaluation ***")
                with torch.no_grad():
                    self.model.eval()
                    eval_results = self.evaluate()
                    self.model.train()

                rank0_print(f"Epoch {epoch+1} evaluation results: {eval_results}")
                
                eval_results = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in eval_results.items()}
                self.log(eval_results)
                del eval_results

                self.model.zero_grad(set_to_none=True)  # Free gradient memory
                torch.cuda.synchronize()
                gc.collect
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

    def second_step_functional(self):
        with torch.no_grad():
            # Determine the maximum sequence length among the accumulated logit gradients.
            max_seq_len = max(grad.shape[1] for grad in self.accumulated_logit_grads)

            # network_fn needed for jvp, could have used autograd grad (maybe should have)
            def network_fn(params, batch):
                return torch.func.functional_call(self.model, params, (), batch).logits

            # dictionary of perturbed params needed for jvp
            perturbed_params = {name: param.detach() for name, param in self.model.named_parameters()}
            # get the batch inputs and logit grads
            
            for batch_cpu, cur_logit_grad in zip(self.accumulated_inputs, self.accumulated_logit_grads):
                cur_logit_grad = cur_logit_grad # .to(self.model.device)

                batch = {}
                for key, value in batch_cpu.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.model.device)
                    else:
                        batch[key] = value
                
                
                batch_size = batch["input_ids"].shape[0]
                
                microbatch_size = 2

                if max_seq_len > 400:
                    microbatch_size = 1
                #microbatch_size = batch_size
                
                #rank0_print(f"Max Seq Len: {max_seq_len}, Microbatch Size: {microbatch_size}")
                model_param_dict = dict(self.model.named_parameters())

                microbatch_count =  (batch_size + microbatch_size - 1) // microbatch_size
                
                # my memory was dying so i had to microbatch, can try other functions
                for i in range(0, batch_size, microbatch_size):
                    microbatch = {k: v[i:i+microbatch_size].detach() for k, v in batch.items()}
                    seq_len = microbatch["input_ids"].shape[1]
                    
                    logit_to_use = cur_logit_grad.contiguous().detach()                    
                    cur_avg_logit_grad = cur_logit_grad[i:i+microbatch_size, :seq_len].contiguous().detach().to(self.model.device)
                    del seq_len, logit_to_use

                    #rank0_print(f"GPU Usage before fJ dtheta fn: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

                    dF_dtheta_fn = torch.func.vjp(lambda theta: network_fn(theta, microbatch), perturbed_params)[1]
                    #rank0_print(f"GPU Usage before vjp grad calculation: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                    vjp_grads = dF_dtheta_fn(cur_avg_logit_grad)[0]
                    del dF_dtheta_fn  # Free the closure immediately.
                    
                    cur_avg_logit_grad.detach().cpu()
                    del cur_avg_logit_grad

                    for key in list(microbatch.keys()):
                        microbatch[key] = microbatch[key].detach().cpu()
                        del key 
                    del microbatch
                    
                    #vjp_grads = dF_dtheta_fn(cur_avg_logit_grad)[0]
                    #rank0_print(f"GPU Usage before geting model param dict: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                    
                    #gc.collect()
                    #torch.cuda.empty_cache()

                    #rank0_print(f"GPU Usage before loading model param dict: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                    # add jvp grads directly into model 
                    #rank0_print(f"GPU Usage before assigning gradients: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                    
                    for name, grad in vjp_grads.items():
                        if grad is not None:
                            param = model_param_dict.get(name)
                            if param is not None:
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param, device=self.model.device)
                                param.grad = param.grad + grad.detach() / microbatch_count
                        vjp_grads[name].detach()
                        del name, grad
                    del vjp_grads, param

                    #rank0_print(f"GPU Usage after assigning gradients: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                
                # delete batch
                for k, v in batch.items():
                    batch[k] = v.detach()
                    del k, v
                del batch, batch_cpu
            del cur_logit_grad, batch_size

            for k, v in perturbed_params.items():
                perturbed_params[k] = v.detach()
                # v.detach()
                del k, v
            del perturbed_params

            for k, v in model_param_dict.items():
                model_param_dict[k] = v.detach()
                del k, v
            del model_param_dict
            
            #print(debug_gpu_variables(locals()))
            
            #rank0_print(f"GPU Usage after second step: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
            del microbatch_size
            del self.accumulated_inputs, self.accumulated_logit_grads
            del max_seq_len

            self.accumulated_inputs = []
            self.accumulated_logit_grads = []
            # model now has functional sam gradients


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

