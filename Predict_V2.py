import torch
import numpy as np
import joblib
from transformers import DistilBertTokenizer, DistilBertModel

# ================= CONFIG =================
MODEL_PATH = "improved_model.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODELS =================
print("Loading improved model...")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.to(DEVICE)
bert_model.eval()

model = joblib.load(MODEL_PATH)


# ================= EMBEDDING FUNCTION =================
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# ================= PREDICTION FUNCTION =================
def predict(text):
    embedding = get_embedding(text)
    cluster = model.predict(embedding)
    return int(cluster[0])


# ================= TEST BLOCK =================
if __name__ == "__main__":
    text = input("Enter news text: ")
    result = predict(text)
    print("Predicted News Cluster:", result)
