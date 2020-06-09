import numpy as np
import math

def k_prac(x1,x2,l):
    '''
    Radial Basis Kernel

    args: 
        x1: 1xd vector
        x2: 1xd vector
        l: lenghtscale
    return:
        int: Similarity measure
    '''
    exponent = - 1/(2*l**2) * np.linalg.norm(x1-x2)**2
    return np.exp(exponent)

def k_homework(x1, x2, l, P=3):
    exponent_1 = - 1/(2*l**2) * np.linalg.norm(x1-x2)**2
    exponent_2 = - 2/(l**2) * math.sin(math.pi * np.linalg.norm(x1-x2) / P )**2
    return np.exp(exponent_1) * np.exp(exponent_2) 

def k_q2(x1,x2,c,d):
    return (x1*x2 +c)**d  
