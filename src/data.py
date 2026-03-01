import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import tiktoken
import itertools

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
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train[:150000]")
        elif split == "test":
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train[150000:150500]")
        else:
            raise ValueError(f"Invalid split: {split}")

        self.context_window = context_window
        self.device = device
        
        self.ds = ds.map(
            self.tokenize,
            batched=True,
            remove_columns=ds.column_names,
        )
    
    def tokenize(self, sample):
        concatenate = lambda prompt, text : " ".join([prompt, text, "<|endoftext|>"])
        texts = [concatenate(sample["prompt"][i], sample["text"][i])  for i in range(len(sample["prompt"]))]
        token_ids = self.tokenizer.encode_batch(texts, allowed_special={"<|endoftext|>"})
        token_ids_flattened = list(itertools.chain.from_iterable(token_ids))
        packed = [
            token_ids_flattened[self.context_window * i : self.context_window * (i + 1)]
            for i in range(len(token_ids_flattened) // self.context_window)
        ]
        return {
            "token_ids": packed,
        }
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return torch.tensor(self.ds[idx]["token_ids"], dtype=torch.long, device=self.device)
    

if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    ds = SmollmDataset(tokenizer, 2048)
    sample = {
        "prompt" : ["Hello, sup witchu" for _ in range(2000)],
        "text" : ["Im alr, sup dude" for _ in range(2000)]
    }
    ds.tokenize(sample)
    
    print(len(ds))
    ds.__getitem__(0)