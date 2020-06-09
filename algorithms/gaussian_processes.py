import numpy as np
import matplotlib.pyplot as plt 
import math
from kernel import k_homework,k_prac,k_q2 

test =      np.loadtxt(r'Datasets\test_inputs_11')
observed =  np.loadtxt(r'Datasets\train_inputs_11')
y =         np.loadtxt(r'Datasets\train_outputs_11')

fig_count = 0


def K(X1,X2,kernel):
    '''Apply a choosen kernel (x1,x2) pairwise on the datasets.

    Args:
        X1 ([n,d]):             input dataset
        X2 ([m,d]):             input dataset
        kernel (method(x1,x2)): a methods which outputs a similarity measure between two vectors

    Returns:
       res ([n,m]):             covariance among the rows of the input datasets
    '''
    res = np.zeros((X1.shape[0], X2.shape[0]))
    for i, p in enumerate(X1):
        for j, q in enumerate(X2):
            res[i,j] = kernel(p,q)
    return res


def sample_prior(X1,N, k):
    '''Uses chosen testponits and a kernel to sample n random multivariate 
    normal distributions from the covariance matrix based on the choosen kernel.

    Args:
        X1 ([n, d]):            test points
        N (int):                number of prior functions
        k (method(x1, x2)):     kernel to create covariance matrix

    Result:
        priors ([N, n]):        prior functions 
    '''
    covar = K(X1,X1,k)
    priors = np.random.multivariate_normal(np.zeros(X1.shape[0]),covar,N)

    return priors


def gausian_function(test, observed, y, k, noise=0.1):
    """Finds the posterior mean and covariance based on the observd training points

    Args:
        test ([n, d]):                      prior test points
        observed ([m, d]):                  observed points
        y (m, 1):                           obeserved targets
        noise (int, optional):              uncertainty of observed points
        k (method(x1, x2)):                 kernel to create covariance matrix

    Returns:
        mu_post ([1, n]):                   posterior mean
        cov_post ([n, n]):                  posterior covaraince 
    """
    K_train = K(observed, observed, k) + noise**2 * np.eye(len(observed)) 
    K_s = K(test, observed, k)          
    K_ss = K(test, test, k)   
    K_train_inv = np.linalg.inv(K_train)        

    # [n, m] * [m, m]^-1 * [m, 1] = [n, 1]
    mu_post = K_s.dot(K_train_inv).dot(y)

    # [n, n] - [n, m] * [m, m]^-1 * [m, n] = [n, n]
    cov_post = K_ss - K_s.dot(K_train_inv).dot(K_s.T)

    return mu_post, cov_post


def new_figure():
    '''Creates a new matplotlib figure'''
    global fig_count
    fig_count += 1
    plt.figure(fig_count)


def plot_priors(priors, min=-5, max=5):
    """Plots the priors on the x-axis defined by min and max

    Args:
        priors ([n, m):          random smapled gaussian functions
        min (int, optional):    x-axis min. Defaults to -5.
        max (int, optional):    x-axis max. Defaults to 5.
    """
    x = np.linspace(min, max, priors.shape[1])

    new_figure()
    plt.title("Gaussian Priors")
    plt.xlabel("Input line")
    plt.ylabel("Sampled value")
    plt.plot(x, priors.T)
    

def plot_gaussian(test, observed, y, k1, n=5):
    """Creates, trains, and plot n gaussian functions in correspondance
    to test points, observed data, and the chosen kernel

    Args:
        test ([n, d]):              chosen test points to evaluate the gaussian
        observed ([m, d]):          observed data
        y ([m, 1]):                 observed targets
        k1 (method(x1, x2)):        similarity kernel
        n (int, optional):          number of gaussian functions
    """
    # Train the gaussian means and covariance on observed points
    mu, covar = gausian_function(test, observed, y, k1)
    samples = np.random.multivariate_normal(mu, covar, n)

    # Plot the results
    new_figure()
    plt.title("Gaussian processes")
    plt.xlabel("Input line")
    plt.ylabel("Prediction")
    plt.plot(test, samples.T)
    plt.plot(observed, y,'.')


# Initialize different kernels
k1 = lambda x1,x2: k_prac(x1,x2,1)
k2 = lambda x1,x2: k_homework(x1,x2,2,3)
k3 = lambda x1,x2: k_q2(x1,x2,5,1)
k4 = lambda x1,x2: k_q2(x1,x2,5,2)
k5 = lambda x1,x2: k_q2(x1,x2,5,3)

"""
p1 = sample_prior(test,5,k1)
p2 = sample_prior(test,5,k2)
p3 = sample_prior(test,5,k3)

plot_priors(p1)
plot_priors(p2)
plot_priors(p3)
"""

# Test how well the kernels collude with the observed data
plot_gaussian(test, observed, y, k1)
plot_gaussian(test, observed, y, k2)
plot_gaussian(test, observed, y, k3)
plot_gaussian(test, observed, y, k4)
plot_gaussian(test, observed, y, k5)

plt.show()

