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

    b. ~~Sliding Window Attention~~

    c. Grouped Query Attention
3. Implement various post-training techniques with simple reward functions
    
    a. RLHF

    b. DPO

    c. GRPO

4. Add features/interesting modules
    
    a. Byte Latent Transformer
    
5. Train on more high quality general datasets