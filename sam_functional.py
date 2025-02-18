import torch


class FunctionalSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FunctionalSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.last_grad_norm = None 
        self.rho = rho
        self.kwargs = kwargs
        self.adaptive = False

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device 
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]),
        p=2)
        return norm

    @torch.no_grad()
    def first_step_functional(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w) 
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def unperturb_functional(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original weights
                p.data = self.state[p]["old_p"]
                del self.state[p]["old_p"]
        if zero_grad:
            self.zero_grad()
    
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


    # this shouldnt be used
    @torch.no_grad()
    def step(self, closure=None):
        assert False, "Do not use step for now"
        assert closure is not None, "SAM optimizer requires a closure, but none was provided"
        closure = torch.enable_grad()(closure)  # ensure gradients are enabled in closure
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
