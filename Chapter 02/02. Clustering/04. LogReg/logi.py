import numpy as np 
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


X, y = make_classification(n_features=2,n_redundant=0, n_informative=2,n_clusters_per_class=1, random_state=1)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()