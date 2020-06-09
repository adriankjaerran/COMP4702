import numpy as np 
import math
import matplotlib.pyplot as plt 
import pandas as pd 

import os

print(os.getcwd())

#df = pd.read_csv(r'7Datasets\pima_indians_diabetes.csv')
df_t = pd.read_csv('datasets/pima_indians_diabetes.csv')

#df_v = pd.read_csv(r'Datasets\BreastCancerValidation.csv',header=None)

x_t = pd.DataFrame.to_numpy(df_t.iloc[:,:-1])
y_t = pd.Series.to_numpy(df_t.iloc[:,-1])


#x_v = pd.DataFrame.to_numpy(df_v.iloc[:,:-1])
#y_v = pd.Series.to_numpy(df_v.iloc[:,-1])



def mu_cov_p(matrix,labels):
    mu = [] # [num_classes,d]
    cov = []  # [num_classes, [d,d]]
    P = []  # [nuk_classes]

    for i,v in enumerate(np.unique(labels)):
        class_i = matrix[labels == v]
        mu.append(np.mean(class_i,0))
        cov.append(np.cov(class_i,rowvar=False))
        P.append(len(class_i)/len(labels))
    
    return mu, cov, P
    


def QDA(x,matrix,labels,mu,cov,P):
    '''
    Maximize class for P(class|xi)

    Args:
        X: The parameter to be classified
        Matrix [N,d]
        Labels [N,1]
        Mean vector [1,d]
        Covaraince Matrix [d,d] - full/shared/diagonal/singular
        Prior probabilities of classes [c]


    Retruns:
       The mean and covariance and weighting of each labels gaussian
       The chances of each 
          
    '''
    class_probability = []
   
    # For each class
    for i in range(len(np.unique(labels))):
        inv_cov = np.linalg.inv(cov[i])
        Wi = -0.5*inv_cov
        wi = inv_cov.dot(mu[i].T)
        wi_0 = -0.5 * mu[i].T.dot(inv_cov).dot(mu[i]) \
               -0.5*math.log(np.linalg.det(cov[i])) + math.log(P[i])
        
        p_x = x.T.dot(Wi).dot(x) + wi.T.dot(x) + wi_0

        class_probability.append(p_x)

    return np.unique(labels)[class_probability.index(max(class_probability))]


def accuaracy(matrix,labels,model):
    '''
    Applies a model to label all the data
    
    Returns:
        Percentage of data correctly classified
    '''
    correct = 0
    for i in range(len(labels)):
        if(model(matrix[i], matrix, labels) == labels[i]):
            correct += 1 
    
    return correct/len(labels)


# Create model
mu,cov,p = mu_cov_p(x_t,y_t)
model = lambda x,matrix,label : QDA(x, matrix, label, mu, cov, p)

#Test model
#error_v = accuaracy(x_v,y_v,model)
error_t = accuaracy(x_t, y_t, model) 

#print('Error validation is {:.2f}'.format(100-error_v*100))
print('Error train is {:.2f}'.format(100-error_t*100))