import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

print("Loading embeddings...")

X = np.load("final_embeddings.npy", allow_pickle=True)

print("Performing KMeans clustering...")

kmeans = KMeans(n_clusters=3, random_state=42)

labels = kmeans.fit_predict(X)

df = pd.read_csv("processed_dataset.csv")

df["label"] = labels

df.to_csv("clustered_dataset.csv", index=False)

print("New dataset with cluster labels saved!")
