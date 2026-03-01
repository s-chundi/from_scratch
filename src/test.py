import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import get_train_test_dataloader
from model import ScratchTransformer, CONTEXT_WINDOW
import wandb
import os
import math
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from wandb_helpers import wandb_log
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8010"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def loss_fn(logits, gt, eot_mask):
    loss = F.cross_entropy(logits, gt, reduction="none")
    loss = loss.masked_fill(eot_mask, 0.0).sum() / (~eot_mask).sum()
    return loss

def validate(rank, world_size, wandb_run, args):
    setup_ddp(rank, world_size)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataloader, test_dataloader = get_train_test_dataloader(tokenizer, CONTEXT_WINDOW, args)
    model = ScratchTransformer(tokenizer).to(rank)
    model.load_state_dict(torch.load(args.checkpoint, map_location={'cuda:%d' % 0: 'cuda:%d' % rank}))
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Eval args: {args}")

    for i, x in enumerate(test_dataloader):
        with torch.no_grad():
            x = x.to(rank)
            y_hat, metadata = model(x[:, :-1])
            eot_mask = (x[:, :-1] == tokenizer.eot_token)
            loss = loss_fn(y_hat.transpose(-1, -2), x[:, 1:], eot_mask)
            wandb_log(wandb_run, rank, {"test/loss" : loss.item()})
    
    for i, x in enumerate(test_dataloader):
        if i == args.num_test_generation:
            break
        with torch.no_grad():
            x = x.to(rank)
            num_generation_tokens = 30
            output = model.module.generate(
                x[:, :-num_generation_tokens], 
                num_tokens=num_generation_tokens
            )
            if rank == 0:
                print(output)
            output_table = wandb.Table(data=output, columns=["Generations"])
            wandb_log(wandb_run, rank, {"generations" : output_table})
    
    torch.cuda.empty_cache()
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_test_generation", type=int, default=20)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    
    settings = wandb.Settings(
        show_errors=True,
        silent=False,
        show_warnings=True,
        show_info=True,
    )
    wandb_run = wandb.init(
        project="Transformer From Scratch",
        settings=settings
    )
    world_size = torch.cuda.device_count()
    mp.spawn(validate, args=(world_size, wandb_run, args), nprocs=world_size)
    wandb.finish()