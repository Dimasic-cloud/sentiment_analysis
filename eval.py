import torch
import pandas as pd
from train import EmotionDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

# download model and tokenizer
tokenizer = BertTokenizer.from_pretrained("best_model")
model = BertForSequenceClassification.from_pretrained("best_model")

# getting device GPU and CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# creating dataloader on test data
test_dataframe = pd.read_csv("test_dataset.csv")
test_emotion, test_text = test_dataframe["Emotion"], test_dataframe["Text"]
test_dataset = EmotionDataset(test_text, test_emotion, tokenizer)
data_colator = DataCollatorWithPadding(tokenizer=tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    collate_fn=data_colator
)

# list for storage prediction and labels
preds = []
labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pred = torch.argmax(outputs.logits, dim=-1)
        preds.extend(pred.cpu().numpy())
        labels.extend(batch["labels"].cpu().numpy())

print("accuracy", accuracy_score(labels, preds))
print("f1", f1_score(labels, preds, average='weighted'))