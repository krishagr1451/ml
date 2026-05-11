import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df.values

scaler = StandardScaler()

X = scaler.fit_transform(X)

model = DBSCAN(eps=1.5, min_samples=5)

clusters = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()