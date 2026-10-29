import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

if __name__ == "__main__":
    embedding = Embedding(
        num_embeddings=4,
        embedding_dim=3,
    )

    token_ids = torch.tensor([
        [2, 0],
        [1, 2],
    ], dtype=torch.long)

    output = embedding(token_ids)

    print("权重形状：", embedding.weight.shape)
    print("输出形状：", output.shape)
    print("参数名称：", list(dict(embedding.named_parameters())))
    print(
        "相同 ID 的向量相同：",
        torch.equal(output[0, 0], output[1, 1]),
    )
