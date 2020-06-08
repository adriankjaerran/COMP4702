import numpy as np 
import math 
import matplotlib.pyplot as plt 
from sklearn import datasets
''' 
Firshers LDA searches for the projection of a dataset which 
maximizes the *between class scatter to within class scatter*
Sb/Sw ratio.

Doesn't need centalization
Supervised
'''

### Load the Dataset) ###

X, y = datasets.load_iris(True)
X, y = np.array(X[:,:2]), np.array(y)


# 1. Calculate mean and class means

mu = np.sum(X,axis=0)/X.shape[0]
mu_c = []
for c in np.unique(y):
  X_c = X[y==c]
  mu_c.append(np.sum(X_c, axis=0)/ X_c.shape[0])


# 2. Compute the scatter between (d x d)
SB = []
for c in np.unique(y):
  num = np.count_nonzero(y == c)
  dif = np.matrix(mu_c[c] - mu)
  SB.append(num * dif.T.dot(dif))
SB = np.sum(SB, axis=0)


# 3. Compute the scatter within clusters (d x d)
SW = []
for c in np.unique(y):
  SW.append((X-mu).T.dot(X-mu))
SW = np.sum(SW,axis=0)


# 4. Compute the eigenvalues and eigenvectors to optimize the equation by the weights
eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(SW),SB))

sort = np.argsort(eigval)[::-1]
eigval, eigvec = eigval[sort], eigvec[sort]

# 5. Transform our data using the two largest eigenvectors
X = X.dot(eigvec[:,:2])

# 6. Plot
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Linear Discriminant Analysis")
plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.plot()
plt.show()

print(np.around(SB,4))



