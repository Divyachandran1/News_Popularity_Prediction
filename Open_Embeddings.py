import numpy as np

data = np.load("final_embeddings.npy", allow_pickle=True)

print("Type of data:", type(data))
print("Shape:", data.shape)
print("First embedding vector:", data[0])
