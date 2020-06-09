import numpy as np 
import math
import matplotlib.pyplot as plt 
import pandas as pd 


df_t = pd.read_csv('datasets/pima_indians_diabetes.csv')

X1 = pd.DataFrame.to_numpy(df_t.iloc[:,:-1])
y1 = pd.Series.to_numpy(df_t.iloc[:,-1])


def mu_cov_P_labels(X,y):
    '''Outputs the mean, covariance and prior class distribution of a dataset

    Args:
        X ([n, d]):         input data
        y ([n, 1]):         labels

    Returns:
        mu ([k,d]):         mean of each class
        cov ([k,[d,d]]):    covaraince of each class
        P ([1,k]):          proior probability of the classes 
        labels ([k])        labels belonging to each class 1..k
    '''
    mu, cov, P, labels = [], [], [], []

    for i,v in enumerate(np.unique(y)):
        class_i = X[y == v]
        mu.append(np.mean(class_i,0))
        cov.append(np.cov(class_i,rowvar=False))
        P.append(len(class_i)/len(y))
        labels.append(v)
    
    return mu, cov, P, labels
    

def qda(x,mu,cov,P,labels=[]):
    '''Implements Quadratic Discriminant Analysis (QDA) as outlined in Alpaydin

    Args:
        x ([1,d]):              instance to be classified
        mu ([k, d]):            mean of each class
        cov ([k, [d,d]]):       covariance of each class
        P ([1, k]):             classes' prior probability
        labels ([k], optinal):  labels belonging to each class 1..k

    Returns: 
        class (str)            index of the class k with highest probability
    '''
    post_prob = []  # Store the probability for each class
    k = len(mu)     # Number of different classes 
    
    # For each class
    for i in range(k):
        # As outlined in Alpaydin 5.20 
        inv_cov = np.linalg.inv(cov[i])
        Wi = -0.5*inv_cov
        wi = inv_cov.dot(mu[i].T)
        wi_0 = -0.5 * mu[i].T.dot(inv_cov).dot(mu[i]) \
               -0.5*math.log(np.linalg.det(cov[i])) + math.log(P[i])
        
        p_x_k = x.T.dot(Wi).dot(x) + wi.T.dot(x) + wi_0 # Probability of x belonging to class k

        post_prob.append(p_x_k)

    index = post_prob.index(max(post_prob))

    if len(labels) == len(mu):
        return labels[index]
    return index


def accuaracy(X1, y1, X2, y2,):
    '''Uses X1, y1 to train a Quadratic Discriminant Analysis model.
    Tests the model on the data X2,y2

    Args:
        X1 ([n,d]):         training data
        y1 ([n,1]):         training labels
        X2 ([m,d]):         test data
        y2 ([m,1]):         test labels

    Retruns:
        accuaracy (float):  percentage of correctly classified test data
    '''
    # Create the QDA-model of the training data
    mu, cov, P, labels = mu_cov_P_labels(X1,y1) 
    model = lambda x: qda(x, mu, cov, P, labels)
    print("Successfully created QDA model")

    # Test the model on the test data
    correct = 0
    for xi,yi in zip(X2,y2):
        if model(xi) == yi:
            correct += 1 
    
    return 100 * correct/len(X2)


#Test model
acc = accuaracy(X1, y1, X1, y1)
print(f'QDA model correctly classified {round(acc,2)}% of the test data.')