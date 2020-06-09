import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.io import loadmat
import pandas as pd

''' 
The principal component Analysis gives us the directions which holds the most
of the variation in the dataset

Works by finding the eigenvectors of the covariance matrix
Unsupervised dimension reduction algorithm 

'''

df = loadmat('Datasets\mnist_train.mat')
y = df.get('train_labels')
X = df.get('train_X')

indexes = [i in [1,3,5,7,9] for i in y]
y = y[indexes]
X = X[indexes]

def pca_plot(X,y=[]):
	'''Produces a 2D-plot unsing the two first dimensions of the data.
	Labels according to y (optinal)

	Args: 
		X ([n,d]): 					input data
		y ([n,1], optinal):			labels
	'''
	title = "PCA plot "
	plt.figure(figsize=(5,3))

	if len(y) == len(X):
		for i,v in enumerate(np.unique(y)):
			indices = np.where(y == v)
			plt.scatter(X[indices,0],X[indices,1],label=v)
		title += 'of classes ' + ' ,'.join([str(k) for k in np.unique(y)])
		plt.legend()
	else:
		plt.scatter(X[:,0], X[:,1])

	plt.title(title)
	plt.xlabel('PCA_1')
	plt.ylabel('PCA_2')
	
	plt.show()



def pca(X,  dim_kept=2, var_kept=0.8,):
	"""Principal Component Analysis (PCA) as outlined in Alpaydin.

	Args:
		X ([n,d]):					input data
		dim_kept (int):				dimension restriction on output
		var_kept (float, optinal): 	variance retained restriction on output
	Returns: 
		out ([n, dim_kept]): 		transformed data
	"""
	# 1. Nomralize the data
	mu = np.sum(X,axis = 0)/X.shape[0]
	std  = np.sqrt(np.sum(np.power(X - mu,2),axis=0) / X.shape[0])
	X = (X-mu)/(std+0.01)

	# 2. Get the eigenvalues and eigenvectors of the covariance matrix
	eigval, eigvec = np.linalg.eig(np.cov(X.T,bias=True,ddof=1))

	# 3. Decicde the output dimension
	cumvar = np.cumsum(eigval / eigval.sum())
	var_restriction = np.searchsorted(cumvar, var_kept) + 1
	out_dim = min(dim_kept, var_restriction)

	# 3. Transform the data
	X = X.dot(eigvec[:,:out_dim])
		
	return X 

X, y = datasets.load_iris(True)
X = pca(X,2)

pca_plot(X,y)