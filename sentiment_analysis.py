import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split


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


# dividing data
emotion_dataset = pd.read_csv("emotion_dataset_raw.csv")
text, emotion = emotion_dataset["Text"], emotion_dataset["Emotion"]

# getting unique emotion
unique_emotion = emotion.unique()
label2id = {label: idx for idx, label in enumerate(unique_emotion)}
id2label = {idx: label for label, idx in label2id.items()}
emotion = emotion.map(label2id)

# devided data into train/val/test
train_text, temp_text, train_emotion, temp_emotion = train_test_split(
    text,
    emotion,
    test_size=0.2,
    random_state=42,
    stratify=emotion
)
val_text, val_emotion, test_text, test_emotion = train_test_split(
    temp_text,
    temp_emotion,
    test_size=0.5,
    random_state=42,
    stratify=temp_emotion
)
