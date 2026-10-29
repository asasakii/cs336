import math
import torch

# 继承 PyTorch 的 Optimizer 基类。基类免费提供三样东西：
# self.param_groups：参数组列表（每组一份超参，允许不同层用不同 lr）
# self.state：default_dict(dict)，以参数张量为 key 存每个参数的历史状态
# add_param_group()、load_state_dict()、zero_grad() 等通用方法
class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9,0.999),
            eps=1e-8,
            weight_decay=0.0,
    ):
        # 检查超参数。
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= betas[1] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        defaults = {
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }
        super().__init__(params,defaults)
        #将 params 交给父类保存。将默认超参数交给父类。初始化参数组和状态管理容器。

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        # closure 是可选的“重新计算损失”的函数。
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("This AdamW implementation does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                p.mul_(1 - lr * weight_decay)

                m.mul_(beta1).add_(g,alpha=1 - beta1)
                v.mul_(beta2).add_(g.square(),alpha=1 - beta2)
                #偏差修正学习率
                step_size = (lr * math.sqrt(1-beta2 ** t) / (1 - beta1 ** t))
                denominator = v.sqrt() + eps
                p.addcdiv_(m,denominator,value=-step_size)
        return loss