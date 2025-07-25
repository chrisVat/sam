import torch
from utils import rank0_print


class FunctionalSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, precondition=False, schedule="constant", schedule_warmup=0.33, rho_min=0.05, **kwargs):
        print("This ran!, precondition is ", precondition)
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FunctionalSAM, self).__init__(params, defaults)
        
        
        # """
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
        print("base optimizer elements: ", self.base_optimizer.__dict__.keys())

        # copy over all base_optimizer elements
        for key, value in self.base_optimizer.__dict__.items():
            #if key not in self.__dict__:
            self.__dict__[key] = value

        #exit()


        self.base_optimizer_type = base_optimizer
        
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
       

        #"""
        self.last_grad_norm = None  # Store the last computed grad norm
        self.precondition = precondition
        self.rho = rho
        self.rho_max = rho
        self.schedule = schedule
        self.schedule_warmup = schedule_warmup
        self.rho_min = rho_min
        self.kwargs = kwargs
        self.device = self.param_groups[0]["params"][0].device
        self.adaptive = False
        self.cur_step = 0
        self.max_steps = 0
        print("Using functional SAM - RHO Is .", self.rho)
        self.has_preallocated = False
        #"""


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
    def precondition_grads(self):        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.base_optimizer.state[p]
                if 'exp_avg_sq' in state:
                    preconditioner = (state['exp_avg_sq'].sqrt() + 1e-6).reciprocal()                
                else:
                    preconditioner = torch.ones_like(p.grad)
                p.grad.mul_(preconditioner)
                
                del preconditioner
        #print("Preconditioned!")


    @torch.no_grad()
    def first_step_functional(self, zero_grad=False, warmup=False): # warmup does nothing here, just so i can reuse files
        if not self.has_preallocated:
            self._preallocate_model()
        
        if self.precondition:
            self.precondition_grads()
        
        grad_norm = self._grad_norm()  # This now stores the grad norm in self.last_grad_norm.
        
        if self.schedule == "linear":
            if self.cur_step < self.schedule_warmup * self.max_steps:
                self.rho = self.rho_max * (1 - (self.cur_step / (self.schedule_warmup * self.max_steps))) + self.rho_min
            else: 
                self.rho = self.rho_max


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
    def restore_old(self, remove_old=False):
        print("restoring old parameters")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if remove_old:
                    p.data.copy_(self.state[p]["old_p"].cuda())
                    del self.state[p]["old_p"]
        self.has_preallocated = False


    @torch.no_grad()
    def final_step(self, restore_old=False):
        if restore_old:
            self.restore_old()
            self.has_preallocated = False



    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure=closure)
    

    # GPU Memory efficiency functions
    @torch.no_grad()
    def _preallocate_model(self):
        #print("preallocating!")
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

    def inspect_optimizer_state(self):
        print("Inspecting optimizer state")
        # iterate over state, check if its a tensor and if its on gpu
        unique_keys = set()

        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if key not in unique_keys:
                        unique_keys.add(key)
                        print(f"Key: {key}, Device: {value.device}")


    @torch.no_grad()
    def move_adamw_second_moment_to_cpu(self, second_only=False):
        target_keys = ["exp_avg_sq", "exp_avg"] if not second_only else ["exp_avg_sq"]

        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            for key in target_keys:  # Iterate over both keys
                if key in state:
                    state[key] = state[key].cpu()  # Move to CPU in a single loop

    @torch.no_grad()
    def move_adamw_second_moment_to_gpu(self, second_only=False):
        target_keys = ["exp_avg_sq", "exp_avg"] if not second_only else ["exp_avg_sq"]
        
        for param in self.base_optimizer.state:
            state = self.base_optimizer.state[param]
            for key in target_keys:  # Iterate over both keys
                if key in state:
                    state[key] = state[key].cuda()  # Move to GPU in a single loop

    @torch.no_grad()
    def move_old_to_gpu(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = self.state[p]["old_p"].cuda()
                del self.state[p]["old_p"]
        self.has_preallocated = True


    # Saving and loading properly
    @torch.no_grad()
    def state_dict_mine(self):
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
    def load_state_dict_mine(self, state_dict):
        #if 'base_optimizer' not in state_dict:
        #    return
        
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
        self.rho = state_dict.get('rho', self.rho)
        self.adaptive = state_dict.get('adaptive', self.adaptive)
        self.kwargs = state_dict.get('kwargs', self.kwargs)
        self.has_preallocated = state_dict.get('has_preallocated', self.has_preallocated)
        self.last_grad_norm = state_dict.get('last_grad_norm', self.last_grad_norm)
        
        if 'base_optimizer' in state_dict:
            print("loading base optimizer!")
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        else:
            self.base_optimizer = self.base_optimizer_type(self.param_groups, **self.kwargs)
            print("creating base optimizer")

        # move base_optimizer to gpu
        self.move_optimizer_to_gpu()

        #self.base_optimizer.param_groups = self.param_groups
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
