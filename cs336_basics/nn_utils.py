import torch
from collections.abc import Iterable
#全局梯度裁剪
@torch.no_grad
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    total_squared_norm = torch.zeros((),device=grads[0].device,dtype=torch.float32,)
    for grad in grads:
        total_squared_norm += grad.float().square().sum()
    total_norm = total_squared_norm.sqrt()
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(scale)
