import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df.values
y = data.target

lda = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X, y)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)

plt.xlabel("LDA 1")
plt.ylabel("LDA 2")

plt.show()