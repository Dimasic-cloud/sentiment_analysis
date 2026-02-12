import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EmotionDataset(Dataset):
    "class for storage and batching data"

    def __init__(self, encodings, labels):
        "inicialization encoding (token from dataset) and labels (emotion from dataset)"
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        "count rows in dataset"
        return len(self.labels)

    def __getitem__(self, idx):
        "getting item by index"
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
