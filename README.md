# An implementation of a Transformer from scratch

Usage
```bash
uv sync
source .venv/bin/activate
python src/pretrain.py
```

Goals:
1. ~~Build a very basic transformer using high level pytorch modules~~

    a. ~~Train on some small dataset and verify sensibility~~

    b. ~~Add lr scheduling, gradient clipping etc.~~

2. Successively break down high level pytorch modules to achieve a more "from scratch" implementation

    a. ~~Multiheaded Attention~~

3. Implement Forward and Backward pass for each module

    a. ~~Linear~~

    b. ~~RMSNorm~~

    c. ~~SILU~~

    d. ~~Embedding~~
    
    e. ~~Multiheaded Attention~~

4. Other things:

    a. KV caching

    b. Padding/Packing

    c. GRPO

    d. Flash Attention
    