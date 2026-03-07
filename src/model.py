import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
import math
import time
from collections import defaultdict
from functional import *

CONTEXT_WINDOW = 8192
    
    
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
            SWIGLUModule(embed_dim, 4 * embed_dim),
            LinearModule(4 * embed_dim, embed_dim),
        )
        
        # breakpoint()
        head_dim = embed_dim // n_head
        angles = 1 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim)) # D_attn / 2
        seq_inds = torch.arange(CONTEXT_WINDOW)[:, None] # S, 1
        pos_angles = angles * seq_inds # S, D_attn / 2
        freqs = torch.polar(torch.ones_like(pos_angles), pos_angles)
        
        self.register_buffer("freqs", freqs, persistent=True)
        self.register_buffer("cache_k", torch.empty(0), persistent=False)
        self.register_buffer("cache_v", torch.empty(0), persistent=False)
    
    def rope_embed(self, x, pos_emb_input):
        """
        x : B, S, Nh, D
        pos_emb_input: B x [0, 1, 2, 0, 1, 2, ...] (S)
        """
        B, S, Nh, Da = x.shape

        x_complex = torch.view_as_complex(x.reshape(B, S, Nh, -1, 2).float())
        x_rotated = x_complex * self.freqs[pos_emb_input][:, :, None, :]
        
        x_out = torch.view_as_real(x_rotated).reshape(B, S, Nh, Da)
        
        return x_out.to(x.dtype)
        
    def forward(self, x, sequence_ids, pos_emb_input, use_cache):
        """
        Args:
            x: B, S, D
            sequence_ids: B x [0, 0, 0, 3, 3, 3, ...] (S)
            pos_emb_input: B x [0, 1, 2, 0, 1, 2, ...] (S)
            use_cache: bool
        """
        ln1x = self.ln1(x)
        qry = self.q_linear(ln1x)
        qry = einops.rearrange(qry, "... sq (nh dattn) -> ... sq nh dattn", nh=self.n_head)
        kv = self.kv_linear(ln1x)
        kv = einops.rearrange(kv, "... sq (twoxnh dattn) -> ... sq twoxnh dattn", twoxnh=self.n_head*2)
        k, v = torch.chunk(kv, 2, dim=-2)
        
        k = self.rope_embed(k, pos_emb_input)
        qry = self.rope_embed(qry, pos_emb_input)
        
        if use_cache:
            if self.cache_k.numel() == 0:
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
        num_blocks=4,
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
        seq = torch.arange(1, S + 1, device=eottoken_mask.device).expand(B, S)
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
        x = self.embed(x)
        pos_emb_input, sequence_ids = self.packing_helper(eottoken_mask)
        # x = x + self.pos_emb(pos_emb_input)
        if use_cache:
            self.cur_pos += x.shape[-2]
        metadata = defaultdict(float)
        for transformer in self.transformers:
            start_time = time.time()
            x = transformer(x, sequence_ids, pos_emb_input, use_cache)
            metadata[f"{transformer.__class__.__name__}_time_ms"] += (time.time() - start_time) * 1000
        x = self.final_ln(x)
        return self.linear(x), metadata
    
    def generate(self, x, num_tokens, temperature = 0.7, p = 0.95):
        def top_p_sampling(out_logits):
            unsorted_out = F.softmax(out_logits / temperature, dim=-1)
            sorted_out, sorted_inds = unsorted_out.sort(dim=-1)
            cumsum_sorted_out = sorted_out.cumsum(dim=-1)
            too_small_mask = cumsum_sorted_out <= (1 - p)
            sorted_out[too_small_mask] = 0
            probs = unsorted_out.scatter(1, sorted_inds, sorted_out)
            return torch.multinomial(probs, 1)
        
        assert num_tokens < self.context_win
        self.reset_cache()

        x = x[:, -self.context_win+num_tokens+1:]
        with torch.no_grad():
            model_out, _ = self.forward(x, use_cache=True)
            inds = top_p_sampling(model_out[:, -1, :])
            x = torch.cat([x, inds], dim=1)
            
            for _ in range(num_tokens - 1):
                model_out, __ = self.forward(x[:, -1:], use_cache=True)
                inds = top_p_sampling(model_out[:, -1, :])
                x = torch.cat([x, inds], dim=1)
                
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
            transformer.cache_k = torch.empty(0)
            transformer.cache_v = torch.empty(0)
            
    
if __name__ == "__main__":
    import tiktoken
    from data import SmollmDataset
    
    tokenizer = tiktoken.get_encoding("gpt2")

    model = ScratchTransformer(tokenizer=tokenizer, num_blocks=2, embed_dim=128, context_win=16)
    # ds = SmollmDataset(tokenizer, 16)

    # token_ids = torch.stack([ds.__getitem__(i) for i in range(3)], dim=0)
    token_ids = torch.randint(tokenizer.n_vocab, size=(3, 15), dtype=torch.long)
    model.generate(token_ids, 3)
    
    logits, metadata = model(token_ids,)
    print(f"Forward: input {token_ids.shape} -> output {logits.shape}")

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, tokenizer.n_vocab), token_ids[:, 1:].reshape(-1))
    loss.backward()
    print(f"Backward: loss={loss.item():.4f}")
    