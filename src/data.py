import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_train_test_dataloader(tokenizer, context_window, args):
    train_dataset = HPDataset(
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
    test_dataset = HPDataset(
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


class HPDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        context_window,
        split="all",
        txt_file="src/harry_potter.txt",
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.tokenizer = tokenizer
        with open(txt_file, "r") as f:
            raw_text = f.read()
            
        train_len = int(len(raw_text) * 0.8)
        if split == "train":
            text = raw_text[:train_len]
        elif split == "test":
            text = raw_text[train_len:]
        elif split == "all":
            text = raw_text
        else:
            raise ValueError("Expected split to be one of ['train', 'test', 'all']")
        
        self.token_ids = self.tokenizer.encode(text)
        self.context_window = context_window
        self.device = device
    def __len__(self):
        return len(self.token_ids) - self.context_window
    
    def __getitem__(self, idx):
        return torch.tensor(self.token_ids[idx:idx + self.context_window + 1], device=self.device)
        