import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import tiktoken

def get_train_test_dataloader(tokenizer, context_window, args, distributed=True):
    train_dataset = SmollmDataset(
        tokenizer,
        context_window=context_window,
        split="train",
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(not distributed),
        pin_memory=True,
        sampler=train_sampler,
    )
    test_dataset = SmollmDataset(
        tokenizer,
        context_window=context_window,
        split="test",
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=True) if distributed else None
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=(not distributed),
        pin_memory=True,
        sampler=test_sampler,
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
        self.tokenizer = tokenizer
        if split == "train":
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train[:500]")
        elif split == "test":
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train[500:550]")
        else:
            raise ValueError(f"Invalid split: {split}")

        self.ds = ds.map(
            self.tokenize,
            batched=True,
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
        return torch.tensor(self.ds[idx]["token_ids"], device=self.device)[:self.context_window]