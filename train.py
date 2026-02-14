import pandas as pd
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
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
device = "cuda" if torch.cuda.is_available() else "CPU"

# field
best_val_loss = float("inf")
epocs = 5

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
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3,
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
    collate_fn=data_collator
)

test_dataset = EmotionDataset(test_text, test_emotion, tokenizer)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)


# params for optim - LLRD
optim_groupp_params = [
    {
        "params": model.classifier.parameters(),
        "lr": 2e-5
    },
]
layer_lrs = {
    11: 1e-5,
    10: 1e-5,
    9: 8e-6,
    8: 8e-6,
    7: 6e-6,
    6: 6e-6,
    5: 4e-6,
    4: 4e-6,
    3: 3e-6,
    2: 3e-6,
    1: 2e-6,
    0: 2e-6,
}
for layer_num, lr in layer_lrs.items():
    optim_groupp_params.append(
        {
            "params": model.bert.encoder.layer[layer_num].parameters(),
            "lr": lr
        }
    )

# optimizer for model
optim = AdamW(optim_groupp_params, weight_decay=0.03)
scaler = GradScaler(device=device)
total_steps = len(train_loader) * epocs
scheduler = get_linear_schedule_with_warmup(
    optimizer=optim,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# fine-tune model
for epoc in range(epocs):

    # train loop
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()

        # autocast - choice float16 or float32
        with autocast(device_type=device):
            outputs = model(**batch)
            loss     = outputs.loss

        scaler.scale(loss).backward()  # counting grad
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)  # gradient clipping
        scaler.step(optim)  # update weights model
        scaler.update()
        scheduler.step()  # change lr
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # saved model and tokenizer
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")