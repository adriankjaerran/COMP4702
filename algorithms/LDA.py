import numpy as np 
import math 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

''' 
Firsher's Linear Discriminant Analysis searches for the projection of a dataset 
which  maximizes the Sb/Sw (between class scatter to within class scatter) ratio.

Doesn't need centalization
Supervised
'''


X, y = datasets.load_iris(True)


def lda(X, y):
	"""Performs Linear Discriminant Analysis on the dataset

	Args:
		X ([n,d]): 	input data
		y ([n,1]):	class labels

	Returns:
		t_X ([n,d]): transformed input data
	"""
	# 1. Find global mean and class labels
	mu = np.sum(X,axis=0)/X.shape[0]
	classes = np.unique(y)

	# 2. Find class mean, scatter between, and scatter within for each class
	mu_c, SB, SW = [], [], []
	for c in classes:
		X_c = X[y==c]
		num_c = len(X_c)

		# Class mean
		mu_c.append(np.sum(X_c, axis=0) / num_c)
 
		# Scatter between (d x d)
		dif = np.matrix(mu_c - mu)
		SB.append(num_c * dif.T.dot(dif))

		# Scatter within cluster (d x d)
		SW.append((X-mu).T.dot(X-mu))

	# 3. Sum up to find the dataset's total SB and SW
	SB = np.sum(SB, axis=0)	
	SW = np.sum(SW, axis=0)

	# 4. Compute the eigenvalues and eigenvectors to optimize the equation by the weights
	eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))

	# 5. Transform our data using the found eigenvectors
	t_X = X.dot(eigvec)

	return t_X


def plot_lda(X, y):
	"""Plots the LDA along the two frist dimensions

	Args:
		X ([n,d]): input data
		y ([n,1]): labels
	"""
	plt.scatter(X[:,0], X[:,1], c=y)
	plt.title("Linear Discriminant Analysis")
	plt.xlabel("LDA1")
	plt.ylabel("LDA2")
	plt.plot()
	plt.show()


transformed_X = lda(X, y)
plot_lda(transformed_X ,y)




