import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ranked_news.csv")

plt.hist(df["predicted_popularity"], bins=20)
plt.title("Popularity Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
