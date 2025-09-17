import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import HPDataset, get_train_test_dataloader
from model import ScratchTransformer, CONTEXT_WINDOW
import wandb
import os
import math
import argparse
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from wandb_helpers import wandb_log, wandb_save_files
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8010"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def validate(model, test_dataloader, rank, wandb_run, args):
    criterion = nn.CrossEntropyLoss()
    for i, x in enumerate(test_dataloader):
        if i == args.num_validation_batches:
            break
        with torch.no_grad():
            x = x.to(rank)
            y_hat, metadata = model(x[:, :-1])
            loss = criterion(y_hat.transpose(-1, -2), x[:, 1:])
            wandb_log(wandb_run, rank, {"test/loss" : loss.item()})
    
    for i, x in enumerate(test_dataloader):
        if i == args.num_test_generation:
            break
        with torch.no_grad():
            x = x.to(rank)
            output = model.module.generate(x, num_tokens=20)
            output_table = wandb.Table(data=output, columns=["Generations"])
            wandb_log(wandb_run, rank, {"generations" : output_table})
    
    torch.cuda.empty_cache()
                
def lr_scheduler(
    optimizer,
    num_cycles=0.5,
    num_warmup_steps=200,
    num_training_steps=6000,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(num_warmup_steps, 1)
        progress = (current_step - num_warmup_steps) / max(num_training_steps - num_warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    
def train(rank, world_size, wandb_run, args): # TODO: make args instance
    setup_ddp(rank, world_size)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataloader, test_dataloader = get_train_test_dataloader(tokenizer, CONTEXT_WINDOW, args)
    model = DDP(ScratchTransformer(tokenizer).to(rank), device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Training args: {args}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = lr_scheduler(optimizer)


    for epoch in range(args.n_epochs):
        for i, x in enumerate(train_dataloader):
            x = x.to(rank)
            y_hat, metadata = model(x[:, :-1])
            loss = criterion(y_hat.transpose(-1, -2), x[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            
            original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_clip)
            optimizer.step()
            scheduler.step()
            wandb_log(
                wandb_run, 
                rank, 
                {
                    "train/loss" : loss.item(), 
                    "train/grad_norm_pre_clip" : original_norm,
                    "lr" : scheduler.get_last_lr()[0],
                    **metadata
                }
            )
                    
            if (i + 1) % args.validate_every == 0:
                validate(model, test_dataloader, rank, wandb_run, args)
    
        if rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.module.state_dict(), f"checkpoints/model_{epoch}.pth")
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--num_test_generation", type=int, default=1)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--num_validation_batches", type=int, default=20)
    parser.add_argument("--norm_clip", type=int, default=1)
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
    wandb_save_files(wandb_run)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, wandb_run, args), nprocs=world_size)
    wandb.finish()