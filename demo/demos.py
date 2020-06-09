import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import datasets 
from scipy.io import loadmat
import algorithms.pca as pca

''' The aim of this python file is to dmeonstrate the algorihmts learned in
COMP4702 Machine Learning University of Queensland 
'''

df = loadmat('Datasets\mnist_train.mat')
y = df.get('train_labels')
X = df.get('train_X')

print(f'y length {len(y)}')
print(f'x length {len(X)}')

indexes = [i in [1,8] for i in y]
y = y[indexes]
X = X[indexes]
#X, y = data['data'][extract], data['labels'][extract]

pca.pca(X)

