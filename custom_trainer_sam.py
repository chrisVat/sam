import torch
from torch.optim import AdamW
from transformers import Trainer, get_scheduler
from tqdm.auto import tqdm
from sam import SAM  
from sam_functional import FunctionalSAM
from utils import rank0_print
import torch.distributed as dist
import datetime
from torch.utils.data.distributed import DistributedSampler
import os


# Trainer specifically for SAM, also designed with aggregation in mind
class FSDPSAMTrainer(Trainer):
    def __init__(self, *args, sam_mode="no", sam_rho=0.05, sam_adaptive=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_mode = sam_mode
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.accumulated_inputs = []
        self.global_step = 0


    def _inner_training_loop(self, *args, **kwargs):
        eval_results = self.evaluate()
        self.log(eval_results)
        rank0_print(f"Initial evaluation results: {eval_results}")
        
        model = self.model
        model.train()
        train_dataloader = self.get_train_dataloader()

        accum_steps = self.args.gradient_accumulation_steps if self.args.gradient_accumulation_steps > 0 else 1
        total_batches = len(train_dataloader)
        total_updates_per_epoch = (total_batches + accum_steps - 1) // accum_steps

        progress_bar = tqdm(total=total_updates_per_epoch * int(self.args.num_train_epochs),
                            desc=f"Rank {dist.get_rank()} Training")

        total_loss = 0.0
        global_step = 0

        for epoch in range(int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            num_batches = 0
            updates_this_epoch = 0
            accumulated_pred_loss = 0.0

            # go over part of minibatch, accumulate inputs 
            # iterate whole batch first to get a better perturbation
            # then reiterate on everything in accumualted_inputs with perturbed model
            for inputs in train_dataloader:
                self.accumulated_inputs.append(inputs)
                prepared_inputs = self._prepare_inputs(inputs)
                with self.autocast_smart_context_manager():
                    loss = self.compute_loss(model, prepared_inputs)
                
                accumulated_pred_loss += loss.item()
                
                loss = loss / accum_steps
                self.accelerator.backward(loss)
                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1

                if len(self.accumulated_inputs) == accum_steps:
                    grad_norm = self.optimizer._grad_norm() # used just for logging
                    # perturb model
                    self.optimizer.first_step(zero_grad=True)

                    sam_loss_sum = 0.0
                    num_accumulated_batches = len(self.accumulated_inputs)
                    # reiterate examples with perturbed model
                    for batch in self.accumulated_inputs:
                        prepared = self._prepare_inputs(batch)
                        with self.autocast_smart_context_manager():
                            loss_second = self.compute_loss(model, prepared)
                        self.accelerator.backward(loss_second)
                        sam_loss_sum += loss_second.item()

                    # undo perturabtion, apply new gradients
                    self.optimizer.second_step(zero_grad=True)
                    self.accumulated_inputs = []

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    global_step += 1
                    updates_this_epoch += 1

                    epoch_float = epoch + (updates_this_epoch / total_updates_per_epoch)

                    logs = {
                        "sam_loss": round(sam_loss_sum / num_accumulated_batches, 4),
                        "pred_loss": round(accumulated_pred_loss / num_accumulated_batches, 4),
                        "avg_epoch_loss": round((epoch_loss / num_batches) * accum_steps, 4),
                        "grad_norm": grad_norm.item() if grad_norm is not None else None,
                        "learning_rate": round(self.lr_scheduler.get_last_lr()[0], 6),
                        "epoch": round(epoch_float, 2),
                    }
                    self.state.global_step = global_step
                    self.callback_handler.on_log(self.args, self.state, self.control, logs)

                    logs['rank'] = dist.get_rank()
                    print(logs)

                    accumulated_pred_loss = 0.0

                    progress_bar.update(1)
                    progress_bar.set_postfix(logs)

            if self.args.eval_strategy == "epoch" and self.eval_dataset is not None:
                print(f"Epoch {epoch+1} finished on rank {dist.get_rank()}. Awaiting evaluation...")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                rank0_print(f"*** Beginning Evaluation ***")
                eval_results = self.evaluate()
                self.log(eval_results)
                rank0_print(f"Epoch {epoch+1} evaluation results: {eval_results}")

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            rank0_print(f"Epoch {epoch+1} finished. Average training loss: {avg_epoch_loss:.4f}")

        progress_bar.close()
        return total_loss

    def create_optimizer(self):
        if self.optimizer is None:
            lr = self.args.learning_rate
            wd = self.args.weight_decay

            def base_optimizer_fn(param_groups):
                return AdamW(param_groups, lr=lr, weight_decay=wd)

            if self.sam_mode == "sam":
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=base_optimizer_fn,
                    rho=self.sam_rho,
                    adaptive=self.sam_adaptive,
                )
            elif self.sam_mode == "prefsam": # this is actually not compatible 
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
