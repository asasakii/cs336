import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        #表示第几对分量
        pair_indices = torch.arange(d_k//2,dtype=torch.float32,device=device)
        inv_freq = 1.0 / (theta**(2*pair_indices/d_k))
        positions = torch.arange(max_seq_len,dtype=torch.float32,device=device)
        angles = positions[:,None] * inv_freq[None,:]
        self.register_buffer("cos_cached",torch.cos(angles),persistent=False)
        self.register_buffer("sin_cached",torch.sin(angles),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor,) -> torch.Tensor:
        cos_values = self.cos_cached[token_positions]
        sin_values = self.sin_cached[token_positions]
        a = x[...,0::2]
        b = x[...,1::2]
        rotated_a = a * cos_values - b * sin_values
        rotated_b = a * sin_values + b * cos_values
        pairs = torch.stack([rotated_a,rotated_b],dim=-1)
        return pairs.flatten(start_dim=-2).to(dtype=x.dtype)
        #[[1,2],[3,4]]->[1,2,3,4]


# [
#     [   # 样本 0，唯一一份供各 head 共用的表
#         [
#             [ 1.0000, 1.0000],   # 第一个 token：位置 0
#             [ 0.5403, 0.9950],   # 第二个 token：位置 1
#             [-0.4161, 0.9801]    # 第三个 token：位置 2
#         ]
#     ],
#
#     [   # 样本 1
#         [
#             [-0.4161, 0.9801],   # 第一个 token：位置 2
#             [ 0.5403, 0.9950],   # 第二个 token：位置 1
#             [ 1.0000, 1.0000]    # 第三个 token：位置 0
#         ]
#     ]
# ]
# cos_values = cos_cached[rope_positions] -> (2,1,3,2)