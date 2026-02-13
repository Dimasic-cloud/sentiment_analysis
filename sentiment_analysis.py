import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split


class EmotionDataset(Dataset):
    "class for storage and batching data"

    def __init__(self, texts, labels, tokenizer):
        "inicialization encoding (token from dataset), labels (emotion from dataset) and tokenizer(data tokenization)"
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        "count rows in dataset"
        return len(self.labels)

    def __getitem__(self, idx):
        "batch tokenization and getting item by index"
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,  # cutting long text
            max_length=128,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# device for processing on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

# dividing data
emotion_dataset = pd.read_csv("emotion_dataset_raw.csv")
text, emotion = emotion_dataset["Text"].tolist(), emotion_dataset["Emotion"]

# getting unique emotion
unique_emotion = emotion.unique()
label2id = {label: idx for idx, label in enumerate(unique_emotion)}
id2label = {idx: label for label, idx in label2id.items()}
emotion = emotion.map(label2id).tolist()

# devided data into train/val/test
train_text, temp_text, train_emotion, temp_emotion = train_test_split(
    text,
    emotion,
    test_size=0.2,
    random_state=42,
    stratify=emotion
)
val_text, test_text, val_emotion, test_emotion = train_test_split(
    temp_text,
    temp_emotion,
    test_size=0.5,
    random_state=42,
    stratify=temp_emotion
)

# tokenizer and bert pretrain
model_name = "bert-base-uncased"
tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)
model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)
model = model.to(device)

# function datacollator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# train loader
train_dataset = EmotionDataset(train_text, train_emotion, tokenizer)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=data_collator
)

val_dataset = EmotionDataset(val_text, val_emotion, tokenizer)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)

test_dataset = EmotionDataset(test_text, test_emotion, tokenizer)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)


# optimizer for model
optim = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# dropout
model.config.hidden_dropout_prob = 0.3

# fine-tune model
for epoc in range(3):

    # train loop
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        outputs = model(**batch)
        loss     = outputs.loss
        loss.backward()
        optim.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # eval loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

        val_loss /= len(val_loader)

    print(f"{epoc}, train loss = {train_loss:.2f}, val loss = {val_loss:.2f}")