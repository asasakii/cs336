import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x:torch.Tensor,dim:int) -> torch.Tensor:
    max_values = x.max(dim=dim,keepdim=True).values
    shifted = x - max_values
    exp_values = torch.exp(shifted)
    denominator = exp_values.sum(dim=dim,keepdim=True)
    return exp_values/denominator

