import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.activations import silu

#例如x（2，4，3）2个序列，每个序列4个token，每个token有3个特征向量
#这里改变的是特征向量的维度
class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        self.w1 = Linear(d_model,d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff,d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model,d_ff, device=device, dtype=dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)

if __name__ == "__main__":
    model = SwiGLU(d_model=3, d_ff=8)
    x = torch.randn(2, 4, 3)

    output = model(x)

    print("输出形状：", output.shape)

    for name, parameter in model.named_parameters():
        print(name, parameter.shape)