import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading embeddings...")

X = np.load("final_embeddings.npy", allow_pickle=True)

df = pd.read_csv("clustered_dataset.csv")

y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training improved model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Validation Accuracy:", accuracy_score(y_test, pred))

with open("improved_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Improved model saved!")
