import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df.values

model = AgglomerativeClustering(n_clusters=3)

clusters = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()