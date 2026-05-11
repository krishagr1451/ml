import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from scipy.cluster.hierarchy import dendrogram, linkage

X, y = make_classification(n_samples=178, n_features=13, n_informative=13, n_redundant=0, n_classes=3, random_state=42)

Z = linkage(X, method='ward')

dendrogram(Z)

plt.show()