import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import tiktoken

def get_train_test_dataloader(tokenizer, context_window, args):
    train_dataset = SmollmDataset(
        tokenizer, 
        context_window=context_window,
        split="train", 
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, shuffle=True),
    )
    test_dataset = SmollmDataset(
        tokenizer, 
        context_window=context_window,
        split="test", 
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(test_dataset, shuffle=True),
    )
    return train_dataloader, test_dataloader


class SmollmDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        context_window,
        split="train",
        device=torch.device("cpu"),
    ):
        super().__init__()
        ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", num_proc=100)
        self.tokenizer = tokenizer
        if split == "train": # TODO: make this dynamic
            ds = ds.select(range(500))
        elif split == "test":
            ds = ds.select(range(500, 600))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.ds = ds.map(
            self.tokenize, 
            batched=True, 
            num_proc=100, 
            remove_columns=["prompt", "text", "token_length", "audience", "format", "seed_data"])

        self.context_window = context_window
        self.device = device
    
    def tokenize(self, sample):
        texts = [sample["prompt"][i] + sample["text"][i] for i in range(len(sample["prompt"]))]
        token_ids = self.tokenizer.encode_batch(texts)
        return {
            "token_ids": token_ids,
        }
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return torch.tensor(self.ds[idx]["token_ids"], device=self.device)