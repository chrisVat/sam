import torch
from torch.optim import AdamW
from transformers import Trainer, get_scheduler
from tqdm.auto import tqdm
import torch.distributed as dist
from sam_functional import FunctionalSAM
from utils import rank0_print
import gc
from transformers.trainer_pt_utils import LabelSmoother
from functorch import vjp
import time


# custom trainer for functional sam designed for accumulating gradients
class FSDPFunctionalSAMTrainer(Trainer):
    def __init__(self, *args, sam_mode="no", sam_rho=0.05, sam_adaptive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_mode = sam_mode
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        self.global_step = 0 

    # accumulates parameter gradients, stores logit gradients
    def get_minibatch_gradients(self, inputs):
        #rank0_print(f"Getting minibatch - Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        prepared_inputs = self._prepare_inputs(inputs)
        self.accumulated_inputs.append(prepared_inputs)  # Save for later perturb pass

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

        self.accelerator.backward(parameter_loss) # hold parameter gradients in model

        logit_grad = logit_grad.detach() 
        self.accumulated_logit_grads.append(logit_grad.detach()) # hold logit gradients for later

        self.total_loss = self.total_loss + parameter_loss.item() 
        self.accumulated_pred_loss = self.accumulated_pred_loss + parameter_loss.item() 

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

        for epoch in range(int(self.args.num_train_epochs)):
            #self.model.train()
            #rank0_print(f"model training: {self.model.training}")

            self.epoch_loss = 0.0
            num_batches = 0
            updates_this_epoch = 0
            self.accumulated_pred_loss = 0.0

            self.accumulated_inputs = []  # stores inputs for second passes
            self.accumulated_logit_grads = [] # stores grads for second pass

            # there are a lot of comments re gpu memory
            # had a lot of problems with memory leaks
            #rank0_print(f"Pre Minibatch - GPU memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
            for inputs in train_dataloader:         
                # Do a forwad pass on a fraction of the mini batch. 
                # Accumulate parameter gradients
                # Store logit gradients in self.accumulated_logit_grads = []
                # I wanted to do a forward pass on the full batch to make sure the perturbation was correct 
                self.get_minibatch_gradients(inputs)

                if len(self.accumulated_inputs) == accum_steps:
                    # perturb the model using parameter gradients
                    self.optimizer.first_step_functional(zero_grad=True)
                    # get g_func_sam
                    with torch.no_grad():
                        self.second_step_functional()              
                    # unperturb model, optimizer step
                    self.optimizer.final_step(zero_grad=True)

                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    num_batches += 1

                    # cleanup is required because we don't just use backward calls
                    del inputs
                    for accum_dict in self.accumulated_inputs:
                        for k, v in accum_dict.items():
                            v.detach()#.cpu()
                            del k, v
                        del accum_dict
                    
                    del self.accumulated_inputs[:]
                    del self.accumulated_logit_grads[:]

                    for grad in self.accumulated_logit_grads:
                        grad.detach()#.cpu()
                        del grad

                    #gc.collect()
                    #torch.cuda.empty_cache()

                    # reset accumulators
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

            # Evaluate at epoch end.
            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                print(f"Epoch {epoch+1} finished on rank {dist.get_rank()}. Awaiting evaluation...")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                    self.model.zero_grad(set_to_none=True)  # Free gradient memory
                    torch.cuda.synchronize()
                    gc.collect()

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

            # should just be fsam, but I started trying to make preconditioned functional
            # ran into bugs and just started with functional sam
            if self.sam_mode == "prefsam":
                self.optimizer = FunctionalSAM(
                    self.model.parameters(),
                    base_optimizer=base_optimizer_fn,
                    lr=lr,
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


    # performs these steps of the paper:
    # # set up the VJP at the perturbed parameters
    # _ , dF_dtheta_fn = vjp ( lambda theta : network_fn ( theta ) , perturbed_params)
    # # do the VJP with the ( unperturbed ) dL_dlogits
    # functional_sam_grad = dF_dtheta_fn ( dL_dlogits ) [0]
    def second_step_functional(self):
        # Determine the maximum sequence length among the accumulated logit gradients.
        max_seq_len = max(grad.shape[1] for grad in self.accumulated_logit_grads)

        # network_fn needed for jvp, could have used autograd grad (maybe should have)
        def network_fn(params, batch):
            return torch.func.functional_call(self.model, params, (), batch).logits

        # dictionary of perturbed params needed for jvp
        perturbed_params = {name: param.detach() for name, param in self.model.named_parameters()}

        with torch.no_grad():
            # get the batch inputs and logit grads
            for batch, cur_logit_grad in zip(self.accumulated_inputs, self.accumulated_logit_grads):
                batch_size = batch["input_ids"].shape[0]
                microbatch_size = 2

                if max_seq_len > 500: # added this to avoid oom - crutch. 
                    microbatch_size = 1
                #rank0_print(f"Max Seq Len: {max_seq_len}, Microbatch Size: {microbatch_size}")
                
                microbatch_count =  (batch_size + microbatch_size - 1) // microbatch_size
                
                # my memory was dying so i had to microbatch, can try other functions
                for i in range(0, batch_size, microbatch_size):
                    microbatch = {k: v[i:i+microbatch_size].detach() for k, v in batch.items()}
                    seq_len = microbatch["input_ids"].shape[1]
                    
                    logit_to_use = cur_logit_grad.contiguous().detach()                    
                    cur_avg_logit_grad = logit_to_use[i:i+microbatch_size, :seq_len].contiguous().detach()
                    del seq_len, logit_to_use

                    dF_dtheta_fn = torch.func.vjp(lambda theta: network_fn(theta, microbatch), perturbed_params)[1]
                    for k, v in microbatch.items():
                        v.detach()
                        del k, v
                    del microbatch
                    
                    vjp_grads = dF_dtheta_fn(cur_avg_logit_grad)[0]
                    del cur_avg_logit_grad, dF_dtheta_fn
                    
                    # add jvp grads directly into model 
                    model_param_dict = dict(self.model.named_parameters())
                    for name, grad in vjp_grads.items():
                        if grad is not None:
                            param = model_param_dict.get(name)
                            if param is not None:
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param, device=self.model.device)
                                param.grad = param.grad + grad.detach() / microbatch_count
                        del name, grad
                    del vjp_grads
            
            del batch, cur_logit_grad, batch_size

        for k, v in perturbed_params.items():
            v.detach()
            del k, v

        # rescale model gradients because we micro batched accumulated gradients
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        param.grad.div_(total_microbatches)
        
        del microbatch_size        
        del self.accumulated_inputs, self.accumulated_logit_grads
        del perturbed_params, max_seq_len

        self.accumulated_inputs = []
        self.accumulated_logit_grads = []
        # model now has functional sam gradients



# debug functions below
def model_grad_l2_norm(model):
    total_norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2) if p.grad is not None else torch.tensor(0.0, device=p.device)
            for p in model.parameters()
        ]), p=2)
    return total_norm.item()


# display all local variables using gpu memory for debugging memory leaks 
def debug_gpu_variables(var_dict, prefix=""):
    # debug_gpu_variables(locals()) -- very useful! 
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


# display all persistent cuda tensors for debugging memory leaks
def find_persistent_cuda_tensors():
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                rank0_print(f"Persistent CUDA Tensor: {obj.shape}, dtype={obj.dtype}, device={obj.device}")
        except:
            pass  

