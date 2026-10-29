from torch import nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            context_length:int,
            d_model:int,
            num_layers:int,
            num_heads:int,
            d_ff:int,
            rope_theta:float,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self,token_ids):
        h = self.token_embeddings(token_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits
