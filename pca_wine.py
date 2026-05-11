import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df.values
y = data.target

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()