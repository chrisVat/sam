import torch



class PreconditionedFunctionalSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(PreconditionedFunctionalSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.last_grad_norm = None  # Store the last computed grad norm
        self.rho = rho
        self.kwargs = kwargs
        self.adaptive = False
        print("Using functional SAM - RHO Is .", self.rho)

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]),
        p=2)
        return norm

    @torch.no_grad()
    def first_step_functional(self, zero_grad=False, warmup=False):
        grad_norm = self._grad_norm()  # This now stores the grad norm in self.last_grad_norm.
        #perturb_sizes = []
        #grad_sizes = []
        #preconditioners = []
        #found_count = 0
        #not_found_count = 0
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                
                state = self.base_optimizer.state[p]
                if 'exp_avg_sq' in state and not warmup:
                    #found_count += 1
                    preconditioner = (state['exp_avg_sq'].sqrt() + 1e-8).reciprocal()
                    #preconditioners.append(torch.norm(preconditioner))
                else:
                    #not_found_count += 1
                    preconditioner = torch.ones_like(p.grad)

                
                """
                if 'custom_exp_avg_sq' in self.state[p]:
                    found_count += 1
                    preconditioner = (self.state[p]['custom_exp_avg_sq'].sqrt() + 1e-3).reciprocal() 
                    #preconditioners.append(torch.norm(preconditioner))
                else:
                    not_found_count += 1
                    preconditioner = torch.ones_like(p.grad)
                    self.state[p]['custom_exp_avg_sq'] = torch.ones_like(p.grad)
                self.state[p]['custom_exp_avg_sq'].mul_(0.9).addcmul_(p.grad, p.grad, value=0.1)
                """

                e_w = p.grad * scale.to(p) * preconditioner
                p.add_(e_w) 
        
                """
                perturb_sizes.append(torch.norm(e_w))
                grad_sizes.append(torch.norm(p.grad))
        # Calculate the average perturbation size
        avg_perturb_size = torch.norm(torch.stack(perturb_sizes), p=2) / len(perturb_sizes)
        avg_grad_size = torch.norm(torch.stack(grad_sizes), p=2) / len(grad_sizes)
        print(f"Average perturbation size: {avg_perturb_size.item()}")
        print(f"Scale: {scale.item()}")
        print(f"Average grad size: {avg_grad_size.item()}")
        """
        if zero_grad:
            self.zero_grad()
        #print(f"Found count: {found_count}, Not found count: {not_found_count}")
        #if len(preconditioners) > 0:
        #    print(f"Average preconditioner size: {sum(preconditioners) / len(preconditioners)}")

    
    @torch.no_grad()
    def final_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original weights
                p.data = self.state[p]["old_p"]
                del self.state[p]["old_p"]
        self.base_optimizer.step()  # update weights using the second gradient
        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM optimizer requires a closure, but none was provided"
        closure = torch.enable_grad()(closure)  # ensure gradients are enabled in closure
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
