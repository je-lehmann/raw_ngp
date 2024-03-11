import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

# provides a truncated version of the exponential function,
# where the input is clamped within a specific range to avoid
# numerical instability during the backward pass

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-80, 80)) # 30

trunc_exp = _trunc_exp.apply