import torch



class FunctionalSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FunctionalSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.base_optimizer_type = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.last_grad_norm = None  # Store the last computed grad norm
        self.rho = rho
        self.kwargs = kwargs
        self.device = self.param_groups[0]["params"][0].device
        self.adaptive = False
        print("Using functional SAM - RHO Is .", self.rho)
        self.has_preallocated = False


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
    def first_step_functional(self, zero_grad=False, warmup=False): # warmup does nothing here, just so i can reuse files
        grad_norm = self._grad_norm()  # This now stores the grad norm in self.last_grad_norm.
        if not self.has_preallocated:
            self._preallocate_model()
        
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"].copy_(p.data) 
                e_w = p.grad * scale.to(p)
                p.add_(e_w) 
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def final_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])

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
    

    # GPU Memory efficiency functions
    @torch.no_grad()
    def _preallocate_model(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = torch.empty_like(p.data, device=p.device)
                self.state[p]["old_p"].copy_(p.data)
        self.has_preallocated = True


    def move_optimizer_to_cpu(self):
        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cpu()


    def move_optimizer_to_gpu(self):
        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cuda()

    @torch.no_grad()
    def move_old_to_cpu(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = self.state[p]["old_p"].cpu()
        self.has_preallocated = False

    @torch.no_grad()
    def move_adamw_second_moment_to_cpu(self):
        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            # exp_avg_sq
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = state["exp_avg_sq"].cpu()

    @torch.no_grad()
    def move_adamw_second_moment_to_gpu(self):
        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            # exp_avg_sq
            if "exp_avg_sq" in state:
                state["exp_avg_sq"] = state["exp_avg_sq"].cuda()

    @torch.no_grad()
    def move_old_to_gpu(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = self.state[p]["old_p"].cuda()
        self.has_preallocated = True

    # Saving and loading properly
    @torch.no_grad()
    def state_dict(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
            'base_optimizer': self.base_optimizer.state_dict(),
            'rho': self.rho,
            'adaptive': self.adaptive,
            'kwargs': self.kwargs,
            'has_preallocated': self.has_preallocated,
            'last_grad_norm': self.last_grad_norm,
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if 'base_optimizer' not in state_dict:
            return
        
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
        self.rho = state_dict.get('rho', self.rho)
        self.adaptive = state_dict.get('adaptive', self.adaptive)
        self.kwargs = state_dict.get('kwargs', self.kwargs)
        self.has_preallocated = state_dict.get('has_preallocated', self.has_preallocated)
        self.last_grad_norm = state_dict.get('last_grad_norm', self.last_grad_norm)
        
        if 'base_optimizer' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        else:
            self.base_optimizer = self.base_optimizer_type(self.param_groups, **self.kwargs)

        # move base_optimizer to gpu
        self.move_optimizer_to_gpu()

        #self.base_optimizer.param_groups = self.param_groups
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

