import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from tqdm import tqdm

# ================= CONFIG ===================

DATA_PATH = "News_dataset.csv"

MODEL_PATH = "improved_model.pkl"

OUTPUT_PATH = "ranked_news.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16

# ============== LOAD COMPONENTS ==============

print("Loading tokenizer and BERT model...")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

bert_model.to(DEVICE)
bert_model.eval()

print("Loading trained popularity model...")
model = joblib.load(MODEL_PATH)

# ============== FUNCTIONS ====================

def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text)


def get_embeddings_batch(text_list):

    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# ============== MAIN PROCESS =================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "Title" in df.columns and "Description" in df.columns:
    df["news_text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
else:
    raise Exception("Dataset must contain Title and Description columns!")

print("Generating embeddings for news articles...")

texts = df["news_text"].apply(preprocess_text).tolist()

embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_emb = get_embeddings_batch(batch_texts)
    embeddings.extend(batch_emb)

X = np.array(embeddings)

print("Predicting popularity scores...")

if hasattr(model, "predict_proba"):
    scores = model.predict_proba(X)[:, 1]
else:
    scores = model.predict(X)

df["predicted_popularity"] = scores

print("Ranking news articles...")

df_sorted = df.sort_values(by="predicted_popularity", ascending=False)

print("Saving ranked results...")

df_sorted.to_csv(OUTPUT_PATH, index=False)

print("\nTop 10 Ranked News:\n")
print(df_sorted[["news_text", "predicted_popularity"]].head(10))

print("\nRanking completed successfully!")
print(f"Results saved at: {OUTPUT_PATH}")
