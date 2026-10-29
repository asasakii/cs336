import torch
from torch import nn
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len:int,theta:float,device=None,dtype=None,use_rope=True,):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.rope = None
        if use_rope:
            self.rope= RotaryPositionalEmbedding(
                theta = theta,
                d_k = self.d_head,
                max_seq_len = max_seq_len,
                device = device,
            )

    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        #x.shape[:-1]：取除最后一维之外的形状。本例为 (1, 3)。
        split_shape = (*x.shape[:-1],self.num_heads,self.d_head)
        q = q.reshape(split_shape).transpose(-3, -2)
        k = k.reshape(split_shape).transpose(-3, -2)
        v = v.reshape(split_shape).transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2],device=x.device)

        #在最后一个维度前插入一个大小为 1的维度
        #RoPE 内部的 token_positions 已经是三维的

            rope_positions = token_positions.unsqueeze(-2)

        # token_positions：  (batch, seq_len)
        # rope_positions：   (batch, 1, seq_len)
        # 取出的cos / sin：   (batch, 1, seq_len, d_head / 2)
        # Q的成对分量：     (batch, heads, seq_len, d_head / 2)

            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        seq_len = x.shape[-2] # token数
        positions = torch.arange(seq_len,device=x.device)
        causal_mask = positions[:,None] >= positions[None,:]
        #（3，1）query  （1，3）key
        head_outputs = scaled_dot_product_attention(Q=q,K=k,V=v,mask=causal_mask)
        #合并多头
        by_token = head_outputs.transpose(-3,-2)
        merged = by_token.flatten(start_dim=-2)
        return self.output_proj(merged)