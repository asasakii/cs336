import torch
from cs336_basics.activations import softmax

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    #transpose交换 K 的最后两个维度
    scores = (Q @ K.transpose(-2,-1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) #~mask 表示对布尔张量逐个取反
    probs = softmax(scores,dim=-1)
    return probs @ V
