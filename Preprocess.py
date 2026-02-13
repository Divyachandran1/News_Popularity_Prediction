import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_csv("News_dataset.csv")

print("Initial shape:", df.shape)

df = df.dropna()

df = df.sample(n=100000, random_state=42)

print("After cleaning:", df.shape)

# Create a simple target label (popularity proxy)
# Since we don't have real popularity, we create synthetic label for demo

df["word_count"] = df["Description"].apply(lambda x: len(str(x).split()))

df["label"] = pd.qcut(df["word_count"], q=3, labels=[0,1,2])

df.to_csv("processed_dataset.csv", index=False)

print("Processed dataset saved as processed_dataset.csv")
