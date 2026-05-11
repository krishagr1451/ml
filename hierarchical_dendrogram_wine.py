import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import dendrogram, linkage

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df.values

Z = linkage(X, method='ward')

dendrogram(Z)

plt.show()