import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
import math
import time
from collections import defaultdict
from functional import *

CONTEXT_WINDOW = 16384
    
    
class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        embed_dim,
        n_head=4,
    ):
        super().__init__()
        assert embed_dim % n_head == 0
        self.ln1 = RMSNormModule(embed_dim)
        self.kv_linear = LinearModule(embed_dim, embed_dim * 2)
        self.q_linear = LinearModule(embed_dim, embed_dim)
        self.n_head = n_head
        self.mlp = nn.Sequential(
            RMSNormModule(embed_dim),
            LinearModule(embed_dim, 4 * embed_dim),
            SILUModule(),
            LinearModule(4 * embed_dim, embed_dim),
        )
        
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        
    def forward(self, x, sequence_ids, use_cache):
        ln1x = self.ln1(x)
        qry = self.q_linear(ln1x)
        qry = einops.rearrange(qry, "... sq (nh dattn) -> ... sq nh dattn", nh=self.n_head)
        kv = self.kv_linear(ln1x)
        kv = einops.rearrange(kv, "... sq (twoxnh dattn) -> ... sq twoxnh dattn", twoxnh=self.n_head*2)
        k, v = torch.chunk(kv, 2, dim=-2)
        
        if use_cache:
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat((self.cache_k, k), dim=1)
                self.cache_v = torch.cat((self.cache_v, v), dim=1)
                k, v = self.cache_k, self.cache_v
                
        out = MHAFunction.apply(qry, k, v, sequence_ids)
        out = einops.rearrange(out, "... sq nh d -> ... sq (nh d)")
        x = out + x
        return self.mlp(x) + x
        
     
class ScratchTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        num_blocks=16,
        embed_dim=640,
        context_win=CONTEXT_WINDOW,
    ):
        super().__init__()
        self.tokenizer=tokenizer
        self.embed_dim = embed_dim
        self.embed = EmbeddingModule(
            num_embeddings=tokenizer.n_vocab, 
            embedding_dim=self.embed_dim, 
            padding_idx=tokenizer.eot_token
        )
        self.context_win = context_win
        self.pos_emb = EmbeddingModule(
            num_embeddings=context_win,
            embedding_dim=embed_dim
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(embed_dim)
                for i in range(num_blocks)
            ]
        )
        self.final_ln = RMSNormModule(embed_dim)
        self.linear = LinearModule(embed_dim, tokenizer.n_vocab)
        
        self.cur_pos = 0
    
    def packing_helper(self, eottoken_mask):
        B, S = eottoken_mask.shape
        seq = torch.arange(1, S + 1).expand(B, S)
        offset = seq * eottoken_mask
        vals, inds = torch.cummax(offset, dim=1)
        sequence_ids = vals - offset
        return seq - sequence_ids - 1 + self.cur_pos, sequence_ids
    
    def forward(self, x, use_cache = False):
        """
        x : Tokens
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] <= self.context_win
        eottoken_mask = x == self.tokenizer.eot_token
        # eottoken_mask = eottoken_mask | (torch.rand(*eottoken_mask.shape) > 0.9)
        x = self.embed(x)
        pos_emb_input, sequence_ids = self.packing_helper(eottoken_mask)
        x = x + self.pos_emb(pos_emb_input)
        if use_cache:
            self.cur_pos += x.shape[-2]
        metadata = defaultdict(float)
        for transformer in self.transformers:
            start_time = time.time()
            x = transformer(x, sequence_ids, use_cache)
            metadata[f"{transformer.__class__.__name__}_time_ms"] += (time.time() - start_time) * 1000
        x = self.final_ln(x)
        return self.linear(x), metadata
        
    def generate(self, x, num_tokens):
        assert num_tokens < self.context_win
        self.reset_cache()

        x = x[:, -self.context_win+num_tokens:]
        with torch.no_grad():
            model_out, _ = self.forward(x, use_cache=True)
            _, inds = torch.max(model_out[:, -1, :], dim=1)
            x = torch.cat([x, inds.unsqueeze(-1)], dim=1)
            
            for _ in range(num_tokens - 1):
                model_out, __ = self.forward(x[:, -1:], use_cache=True)
                __, inds = torch.max(model_out[:, -1, :], dim=1)
                x = torch.cat([x, inds.unsqueeze(-1)], dim=1)
                
            out_texts = []
            
            for batch_idx in range(x.shape[0]):
                in_string = self.tokenizer.decode(x[batch_idx, -150:-num_tokens].tolist())
                out_string = self.tokenizer.decode(x[batch_idx, -num_tokens:].tolist())
                out_texts.append([f"Input:\n...{in_string}\nOutput:\n{out_string}"])
            
            self.reset_cache()
            return out_texts
        
    def reset_cache(self):
        self.cur_pos = 0
        for transformer in self.transformers:
            transformer.cache_k = None
            transformer.cache_v = None
            
    
if __name__ == "__main__":
    import tiktoken
    from data import SmollmDataset
    
    tokenizer = tiktoken.get_encoding("gpt2")

    model = ScratchTransformer(tokenizer=tokenizer, num_blocks=2, embed_dim=128, context_win=16)
    ds = SmollmDataset(tokenizer, 16)

    token_ids = torch.stack([ds.__getitem__(i) for i in range(3)], dim=0)
    model.generate(token_ids, 3)
    
    logits, metadata = model(token_ids, )
    print(f"Forward: input {token_ids.shape} -> output {logits.shape}")

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, tokenizer.n_vocab), token_ids[:, 1:].reshape(-1))
    loss.backward()
    print(f"Backward: loss={loss.item():.4f}")
    