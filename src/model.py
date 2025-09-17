import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
import math
import time
from collections import defaultdict

CONTEXT_WINDOW = 8192

class CustomTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        window_size=0,
        n_head=4,
    ):
        super().__init__()
        assert embed_dim % n_head == 0
        self.ln1 = nn.LayerNorm(embed_dim)
        self.kv_linear = nn.Linear(embed_dim, embed_dim * 2)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.window_size = window_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        
    def forward(self, x, key_pad_mask):
        ln1x = self.ln1(x)
        qry = self.q_linear(ln1x)
        qry = einops.rearrange(qry, "... sq (nh dattn) -> ... sq nh dattn", nh=self.n_head)
        kv = self.kv_linear(ln1x[:, -self.window_size:, :])
        kv = einops.rearrange(kv, "... sq (twoxnh dattn) -> ... sq twoxnh dattn", twoxnh=self.n_head*2)
        k, v = torch.chunk(kv, 2, dim=-2)
        attn_scores = einops.einsum(qry, k, "... sq nh d, ... sk nh d -> ... nh sq sk") / math.sqrt(x.shape[-1])
        causal_attn_mask = torch.triu(
            torch.ones(
                1,
                qry.shape[1],
                k.shape[1],
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )
        
        attn_scores.masked_fill_(key_pad_mask[:, None, None, -self.window_size:], -1e10)
        attn_scores.masked_fill_(causal_attn_mask, -1e10)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = einops.einsum(attn_weights, v, "... nh sq sk, ... sk nh d -> ... sq nh d")
        out = einops.rearrange(out, "... sq nh d -> ... sq (nh d)")
        return self.mlp(out) + x
        
        
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads=8,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        # Not doing dropout b/c its pretty useless for large pretraining datasets
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        
    
    def forward(self, x, key_pad_mask):
        ln1x = self.ln1(x)
        key = self.Wk(ln1x)
        qry = self.Wq(ln1x)
        val = self.Wv(ln1x)
        attn_mask = torch.triu(
            torch.ones(
                (qry.shape[1], key.shape[1]), 
                dtype=torch.bool,
                device=x.device
            ), 
            diagonal=1
        )
            
        attnx, _ = self.mha(
            qry, key, val, 
            key_padding_mask=key_pad_mask, 
            attn_mask=attn_mask, 
            need_weights=False
        )
        residx = attnx + x
        ln2x = self.ln2(residx)
        return self.mlp(ln2x) + residx
    
        
class ScratchTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        num_blocks=8,
        embed_dim=512,
        context_win=CONTEXT_WINDOW,
        experimental=False,
    ):
        super().__init__()
        self.tokenizer=tokenizer
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(
            num_embeddings=tokenizer.n_vocab, 
            embedding_dim=self.embed_dim, 
            padding_idx=tokenizer.eot_token
        )
        self.experimental = experimental
        self.context_win = CONTEXT_WINDOW
        self.pos_emb = nn.Embedding(
            num_embeddings=context_win,
            embedding_dim=embed_dim
        )
        self.transformers = nn.ModuleList(
            [
                CustomTransformerBlock(
                    embed_dim, 
                    window_size=2000
                )
                if i % 2 == 0 else TransformerBlock(embed_dim)
                for i in range(num_blocks)
            ]
        )
        self.final_ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, tokenizer.n_vocab)
        
    def forward(self, x):
        """
        x : Tokens
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        key_pad_mask = x == self.tokenizer.eot_token
        x = self.embed(x)
        self.pos_emb(torch.arange(x.shape[-2], device=x.device))
        metadata = defaultdict(float)
        for transformer in self.transformers:
            start_time = time.time()
            x = transformer(x, key_pad_mask)
            metadata[f"{transformer.__class__.__name__}_time_ms"] += (time.time() - start_time) * 1000
        x = self.final_ln(x)
        return self.linear(x), metadata
    
    def generate(self, x, num_tokens):
        for _ in range(num_tokens):
            model_out, __ = self.forward(x[:, -CONTEXT_WINDOW:])
            __, inds = torch.max(model_out[:, -1, :], dim=1)
            x = torch.cat([x, inds.unsqueeze(-1)], dim=1)
        
        out_texts = []
        for batch_idx in range(x.shape[0]):
            out_string = self.tokenizer.decode(x[batch_idx, -100:].tolist())
            out_texts.append([out_string])
        return out_texts