import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math

class LinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inp, weight, bias):
        ctx.save_for_backward(inp, weight)
        
        return inp @ weight + bias
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        inp, weight = ctx.saved_tensors
        
        din, dweight, dbias = None, None, None
        
        if ctx.needs_input_grad[0]:
            din = einops.einsum(
                dout, weight, "... out, inp out -> ... inp"
            )
        if ctx.needs_input_grad[1]:
            dweight = einops.einsum(
                dout, inp, "... out, ... inp -> inp out"
            )
        if ctx.needs_input_grad[2]:
            dbias = einops.reduce(dout, "... out -> 1 out", "sum")
            
        return din, dweight, dbias
    
class LinearModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim
    ):
        super().__init__()
        self.input_dim = input_dim
        
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(1, output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        scale = 1 / math.sqrt(self.input_dim)
        nn.init.uniform_(self.weight, -scale, scale)
        nn.init.uniform_(self.bias, -scale, scale)
        
    def forward(self, inp):
        return LinearFunction.apply(inp, self.weight, self.bias)
    
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inp, gamma):
        rms = torch.sqrt(1e-8 + (inp ** 2).mean(dim=-1, keepdim=True))
        ctx.save_for_backward(inp, gamma, rms)
        return gamma * (inp / rms)
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        inp, gamma, rms = ctx.saved_tensors
        x_norm = inp / rms
        grad_x_norm = dout * gamma
        
        dinp, dgamma = None, None
        
        if ctx.needs_input_grad[0]:
            dinp = (
                grad_x_norm - x_norm * (x_norm * grad_x_norm).mean(dim=-1, keepdim=True)
            ) / rms
        if ctx.needs_input_grad[1]:
            dgamma = einops.reduce(dout * x_norm, "... embed_dim -> embed_dim", "sum")
            
        return dinp, dgamma
    
class RMSNormModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim,))
        
    def forward(self, inp):
        return RMSNormFunction.apply(inp, self.gamma)

class SILUFunction(torch.autograd.Function):
    """[Deprecated] Replaced by SWIGLU"""
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        # Assumes linear already happened
        return inp * 1 / (1 + torch.exp(-inp))
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        inp, = ctx.saved_tensors
        sigmoid_x = 1 / (1 + torch.exp(-inp))

        if ctx.needs_input_grad[0]:
            return dout * sigmoid_x * (1 + inp * (1 - sigmoid_x))
        else:
            return None

class SILUModule(nn.Module):
    """Replaced by SWIGLU"""
    def __init__(self):
        super().__init__()
        
    def forward(self, inp):
        return SILUFunction.apply(inp)

class SWIGLUFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inp, w1, b1, w2, b2):
        u = inp @ w1 + b1
        v = inp @ w2 + b2
        ctx.save_for_backward(inp, w1, b1, w2, b2)
        
        return u * torch.sigmoid(u) * v
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        inp, w1, b1, w2, b2 = ctx.saved_tensors
        
        u = inp @ w1 + b1
        sigmoid_u = torch.sigmoid(u)    
        v = inp @ w2 + b2
        
        dout_dsilu = dout * v
        dsilu_du = sigmoid_u * (1 + u * (1 - sigmoid_u))
        dout_du = dout_dsilu * dsilu_du
        dout_dv = dout * u * sigmoid_u
        
        dinp, dw1, db1, dw2, db2 = None, None, None, None, None
        if ctx.needs_input_grad[0]:
            dout_v_dinp = einops.einsum(
                dout_dv, w2, "... out, inp out -> ... inp"
            )
            dout_u_dinp = einops.einsum(
                dout_du, w1, "... out, inp out -> ... inp"
            )
            dinp = dout_v_dinp + dout_u_dinp
        if ctx.needs_input_grad[1]:
            dw1 = einops.einsum(
                dout_du, inp, "... out, ... inp -> inp out"
            )
        if ctx.needs_input_grad[2]:
            db1 = einops.reduce(
                dout_du, "... out -> 1 out", "sum"
            )
        if ctx.needs_input_grad[3]:
            dw2 = einops.einsum(
                dout_dv, inp, "... out, ... inp -> inp out"
            )
        if ctx.needs_input_grad[4]:
            db2 = einops.reduce(
                dout_dv, "... out -> 1 out", "sum"
            )
            
        return dinp, dw1, db1, dw2, db2

        
class SWIGLUModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w1 = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b1 = nn.Parameter(torch.empty(1, output_dim))
        self.w2 = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b2 = nn.Parameter(torch.empty(1, output_dim))

        self.reset_parameters()
        
    def reset_parameters(self):
        scale = 1 / math.sqrt(self.input_dim)
        for params in [self.w1, self.b1, self.w2, self.b2]:
            nn.init.uniform_(params, -scale, scale)
        
    def forward(self, inp):
        return SWIGLUFunction.apply(inp, self.w1, self.b1, self.w2, self.b2)

class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, token_ids, emb_matrix):
        """
        Args: 
            token_ids: longtensor of shape (B, S)
            emb_matrix: weight matrix of shape (n_vocab, D)
        Returns:
            tensor of shape B, S, D
        """
        ctx.save_for_backward(token_ids, emb_matrix)
        return emb_matrix[token_ids, :]

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        token_ids, emb_matrix = ctx.saved_tensors
        nv, D = emb_matrix.shape
        demb = None
        if ctx.needs_input_grad[1]:
            demb = torch.zeros_like(emb_matrix)
            grads_flattened = dout.reshape(-1, D)
            demb.index_add_(0, token_ids.reshape(-1), grads_flattened)
            
        return None, demb
    
class EmbeddingModule(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        scale = 1 / math.sqrt(self.embed_dim)
        nn.init.uniform_(self.weight, -scale, scale)
        
    def forward(self, inp):
        return EmbeddingFunction.apply(inp, self.weight)
    
class MHAFunction(torch.autograd.Function):
    @staticmethod
    def compute_attn_weights(qry, k, v, sequence_ids):
        attn_scores = einops.einsum(qry, k, "... sq nh d, ... sk nh d -> ... nh sq sk") / math.sqrt(k.shape[-1])
        causal_attn_mask = torch.triu(
            torch.ones(
                qry.shape[1],
                k.shape[1],
                dtype=torch.bool,
                device=k.device
            ),
            diagonal=k.shape[1] - qry.shape[1] + 1
        )
        packing_mask = sequence_ids[:, None, None, :] != sequence_ids[:, None, :, None]
        attn_scores.masked_fill_(packing_mask, -1e10)
        attn_scores.masked_fill_(causal_attn_mask, -1e10)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights
    
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, qry, k, v, sequence_ids):
        """
        Args:
            Q, K, V with shape B, S, Nh, D_attn
            key_pad_mask will be used in the future
            
        Returns:
            tensor with shape B, S, Nh, D_attn
        """
        attn_weights = MHAFunction.compute_attn_weights(qry, k, v, sequence_ids)
        out = einops.einsum(attn_weights, v, "... nh sq sk, ... sk nh d -> ... sq nh d")
        
        ctx.save_for_backward(v, k, qry, sequence_ids)
        return out
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        v, k, qry, sequence_ids = ctx.saved_tensors
        attn_weights = MHAFunction.compute_attn_weights(qry, k, v, sequence_ids)
        
        dim = v.shape[-1]
        dqry, dk, dv, dkpm = None, None, None, None
        
        if ctx.needs_input_grad[2]:
            dv = einops.einsum(
                attn_weights, dout,
                "... nh sq sk, ... sq nh d -> ... sk nh d",
            )
            
        dattn_weights = einops.einsum(
            v, dout,
            "... sk nh d, ... sq nh d -> ... nh sq sk",
        )
        dlss = einops.einsum(
            attn_weights, dattn_weights, 
            "... sq sk, ... sq sk -> ... sq",
        ).unsqueeze(dim=-1)
        dattn_scores = attn_weights * (dattn_weights - dlss)
        if ctx.needs_input_grad[0]:
            dqry = einops.einsum(
                dattn_scores, k, 
                "... nh sq sk, ... sk nh d -> ... sq nh d"
            ) / math.sqrt(dim)
        if ctx.needs_input_grad[1]:
            dk = einops.einsum(
                dattn_scores, qry, 
                "... nh sq sk, ... sq nh d -> ... sk nh d"
            ) / math.sqrt(dim)
            
        return dqry, dk, dv, dkpm