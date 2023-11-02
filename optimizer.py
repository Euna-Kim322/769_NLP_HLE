from typing import Callable, Iterable, Tuple
import math
import torch
from torch.optim import Optimizer


#https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3, #learning rate
            betas: Tuple[float, float] = (0.9, 0.999), #coefficients used for computing runing ave of gradient and its square. 
            eps: float = 1e-6,#in pytorch, 1e-8
            weight_decay: float = 0.0, # L2 penalty
            correct_bias: bool = True, 
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        
     ### Implemented ###   
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]: 
                if p.grad is None:  # while parameter doesn't converged. 
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                                        
                #1. State should be stored in this dictionary
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data) #initialize 1st moment vec
                    state["v"] = torch.zeros_like(p.data) #initial 2nd moment vec
                    state["step"] = 0 # initialize timestep
                    
                #2. Access hyperparameters from the `group` dictionary : Required
                alpha = group["lr"] # stepsize
                beta1, beta2 = group["betas"] # exponential decay rates for the moment estimates
                eps = group["eps"] # Initial parameter vector
                weight_decay = group["weight_decay"] # 
                
                #correct_bias = group["correct_bias"] ## bool = True

                #3. Update first and second moments of the gradients
                state["m"].mul_(beta1).add_((1 - beta1) * grad) # Update biased 1st mom estimate
                state["v"].mul_(beta2).add_((1 - beta2) * grad * grad) # Update biased 2nd mom esitmate

                state["step"] += 1  # t = t+1

            
                #4. Bias correction
                m_hat = state["m"] / (1 - beta1 ** state["step"]) # Compute bias-corrected 1st moment estimate
                v_hat = state["v"] / (1 - beta2 ** state["step"]) # Compute 2nd moment estimate

                # 5. Update parameters (addcdiv : p.data = p.data + value *(tensor1/tensor2) / p.data + (-alpha)*(m_hat)/(v_hat.sqrt()+eps)
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-alpha)
                
                
                # 6.Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if weight_decay != 0: # 
                    p.data.add_(p.data, alpha=-alpha * weight_decay)
                    
        return loss
    

    