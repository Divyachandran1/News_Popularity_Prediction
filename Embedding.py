import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import os

# ================= CONFIG =================
DATA_PATH = "processed_dataset.csv"
EMBEDDING_OUTPUT = "final_embeddings.npy"
CHECKPOINT_FILE = "embedding_checkpoint.npy"
BATCH_SIZE = 16
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

# Combine title + description
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
texts = df["text"].tolist()

print(f"Total articles: {len(texts)}")

# ================= LOAD MODEL =================
print("Loading DistilBERT model...")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

model.to(DEVICE)
model.eval()

# ================= EMBEDDING FUNCTION =================
def generate_embeddings(text_list):

    all_embeddings = []

    for i in tqdm(range(0, len(text_list), BATCH_SIZE)):

        batch_texts = text_list[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

        # Save checkpoint every 5000 samples
        if (i // BATCH_SIZE) % 300 == 0:
            np.save(CHECKPOINT_FILE, np.vstack(all_embeddings))

    return np.vstack(all_embeddings)

# ================= RUN =================
print("Generating embeddings...")

final_embeddings = generate_embeddings(texts)

# Save final embeddings
np.save(EMBEDDING_OUTPUT, final_embeddings)

print("Embedding generation complete.")
print(f"Saved to: {EMBEDDING_OUTPUT}")
