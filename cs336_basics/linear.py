import math
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None,):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype,))

        std = math.sqrt(2.0/(in_features+out_features))

        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std,b=3*std,)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

if __name__ == "__main__":
    layer = Linear(in_features=3, out_features=2)

    # 使用固定权重，方便手算核对。
    layer.load_state_dict({
        "weight": torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
    })

    x = torch.tensor([[1.0, 2.0, 3.0]])
    print("计算结果：", layer(x))

    batch_x = torch.ones(2, 4, 3)
    print("批量输出形状：", layer(batch_x).shape)

    print("参数名称：", list(dict(layer.named_parameters())))