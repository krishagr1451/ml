import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df.values

model = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()