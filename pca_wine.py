import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

X, y = make_classification(n_samples=178, n_features=13, n_informative=13, n_redundant=0, n_classes=3, random_state=42)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()