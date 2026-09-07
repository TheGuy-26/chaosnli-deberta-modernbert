"""Quick check that ModernBERT loads and returns 3-way NLI probs."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "tasksource/ModernBERT-large-nli"
print("Loading", name)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
model.eval()

pre = "Sorry but that's how it is."
hyp = "This is how things are and there are no apologies about it."
inputs = tokenizer(pre, hyp, return_tensors="pt", truncation=True)

with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)

print(probs.tolist()[0])
print(model.config.id2label)
