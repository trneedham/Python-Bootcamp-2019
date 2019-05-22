"""
Algorithm for kMeans clustering
"""

import numpy as np

def cluster_centers(X,K):
    return X[np.random.choice(len(X),size=K)]

def closest(X, centers):
    distances = np.linalg.norm(X - centers[:, np.newaxis,:], axis=2)
    return np.argmin(distances, axis=0)

def new_centers(X, centers):
    c = closest(X, centers)
    K = len(np.unique(c)) # Determine K by finding the number of labels in c
    return np.array([X[c==k].mean(axis=0) for k in range(K)])

def kMeans(X, K, max_iter = 10000):
    # Initializations
    centers = X[np.random.choice(len(X),size=K)]
    iteration = 0
    Delta = 1
    # While loop with a hard limit on number of iterations
    while Delta > .001 and iteration < max_iter:
        moved = new_centers(X,centers)
        Delta = np.linalg.norm( moved - centers )
        iteration = iteration+1
        centers = moved
    print('Iterations to converge: ', iteration)
    labels = closest(X,centers)
    # Output is a tuple
    return centers, labels
