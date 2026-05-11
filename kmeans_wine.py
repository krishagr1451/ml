import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

X, y = make_classification(n_samples=178, n_features=13, n_informative=13, n_redundant=0, n_classes=3, random_state=42)

model = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()