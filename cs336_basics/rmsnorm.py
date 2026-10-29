import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.square().mean(dim=-1,keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        output = normalized * self.weight
        return output.to(in_dtype)

if __name__ == "__main__":
    norm = RMSNorm(d_model=2)

    x = torch.tensor([
        [[3.0, 4.0], [0.0, 0.0]]
    ])

    output = norm(x)

    print("输出：", output)
    print("输出形状：", output.shape)
    print("参数初值：", norm.weight.detach())
    print("全部有限：", torch.isfinite(output).all().item())

    x_half = x.to(torch.float16)
    print("低精度输入的输出类型：", norm(x_half).dtype)