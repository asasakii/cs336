import torch
from torch import nn

from cs336_basics.multihead_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int,
            d_ff:int,
            max_seq_len:int,
            theta:float,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model,num_heads,max_seq_len,theta,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn = SwiGLU(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor :
        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        return z