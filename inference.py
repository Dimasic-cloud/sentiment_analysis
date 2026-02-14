import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

tokenizer = BertTokenizer.from_pretrained("best_model")
model = BertForSequenceClassification.from_pretrained("best_model")

device = "cuda" if torch.cuda.is_available() else "cpu"
preds = []
labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pred = torch.argmax(outputs.logits, dim=-1)
        preds.extend(pred.cpu().numpy())
        labels.extend(batch["labels"].cpu().numpy())

print("accuracy", accuracy_score(labels, preds))
print("f1", f1_score(labels, preds))