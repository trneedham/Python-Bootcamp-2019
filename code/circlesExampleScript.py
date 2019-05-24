"""
Generate the 'concentric circles' example for clustering experiments
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X1, y1 = make_circles(n_samples=50, noise = 0.02, random_state = 3)
plt.scatter(X1[:,0],X1[:,1],c=y1)
plt.show()
