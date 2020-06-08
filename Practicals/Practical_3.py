import numpy as np 
import matplotlib.pyplot as plt 
import math
import pandas as pd 

''' Multivariate Parametric Techniques'''


# Question 1 

''' 8 input features and 768 data points'''
''' Apply and evaluate quadratic disciminant analysis on the diabetes dataset ''' 

# Question 4

df = pd.read_csv(r'Datasets\iris.txt', sep=',')

matrix = pd.Series.to_numpy(df.iloc[:,3])
plt.hist(matrix,500)

plt.show()


