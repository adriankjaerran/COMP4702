import numpy as np 
import math 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats 
### Prepare datasets ### 

def func1(x):
    return 2*np.sin(1.5*x)

def func2(x):
    return func1(x) + np.random.normal(size=x.shape)

x = np.linspace(-5,5,100)
#print(func1(x))
#acctual = func1(x)
#training = func2(x)
#test = func2(x)
#plt.plot(x,acctual)
#
#plt.scatter(x, func2(x),c='red',marker='.')


# Question 1. 

#'''
#-Fit polynomials up to order 9
#-Plot the error for each polynomial
#'''
#def SSE(x1,x2):
#    return np.linalg.norm(x1-x2,2)**2
#
#def find_polynomial_error(x1,x2,d):
#    coeff_matrix  = []
#    for i in range(d):
#        coeff_matrix.append(np.polyfit(x1,x2, i+1))
#
#    error = []
#
#    for i in range(d):
#        error.append(SSE(x2, np.polyval(coeff_matrix[i], x1) ))
#
#    return error
#
#plt.figure(2)
#plt.plot(range(1,9), find_polynomial_error(x,trainig,9))
#
## Question 2
#housing = np.loadtxt(r'Datasets\housing.data')
#plt.scatter(housing[5], housing[13])
#plt.figure(3)
#plt.plot(range(1,10), find_polynomial_error(housing[5],housing[13],9))
#plt.show()


# Question 4
'''
Create Gaussian distributions for each of the classes
Create a plot of how the class porbability for each column
'''
df = pd.read_csv(r'Datasets\iris.txt')
mapping = {'Iris-setosa' : 0}
df = df.replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])

iris = pd.DataFrame.to_numpy(df.iloc[:,[0,-1]])

class0 = iris[iris[:,-1] == 0] 
class1 = iris[iris[:,-1] == 1]
class2 = iris[iris[:,-1] == 2]

mean0,std0 = class0[:,0].mean(), class0[:,0].std()
mean1,std1 = class1[:,0].mean(), class1[:,0].std()
mean2,std2 = class2[:,0].mean(), class2[:,0].std()

minimum = pd.Series.min(df.iloc[:,0])
maximum = pd.Series.max(df.iloc[:,0])
x = np.linspace(0,200,100)

p0 = stats.norm.pdf(x,mean0,std0)
p1 = stats.norm.pdf(x,mean1,std1)
p2 = stats.norm.pdf(x,mean2,std2)

Prob = [p0,p1,p2]
Prob = [ flower/sum(Prob) for flower in Prob]

plt.plot(x,np.array(Prob).T)
plt.show()


print(mean1, std1)


