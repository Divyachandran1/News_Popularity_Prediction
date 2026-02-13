import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

X = np.load("final_embeddings.npy", allow_pickle=True)
df = pd.read_csv("clustered_dataset.csv")

y = df["label"].values

with open("improved_model.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(X)

print("\nClassification Report:\n")
print(classification_report(y, pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y, pred))
