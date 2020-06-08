import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd



# Question 3 
df = np.loadtxt(r'Datasets\prac1_q3.dat')
std = np.std(df)
mean = np.mean(df)
print(f'Mean: {mean}  Std: {std}')


# Question 4 
'''Given the dataset "prac1_q4_dat"
plot the first column against the second column as blue circles
plot the third column against the fourth column as red squares
'''
#df = np.loadtxt(r'Datasets\prac1_q4.dat')
#print(df.shape)
#plt.scatter(df[:,0],df[:,1], c='blue',marker='o')
#plt.scatter(df[:,2],df[:,3], c='red',marker='s')
#plt.xlabel('Inputs')
#plt.ylabel('Outputs')
#plt.title('Question 4')
#
#plt.show()



# Question 5 
#
#plt.hist(np.random.normal(2,4,(1000)),30)
#plt.xlabel('Random Variable')
#plt.ylabel('Frequency')
#plt.title('Qustion 5')
#plt.show()
#
def q6(vector, n):
    """
    Revertse the input vector in chunks of size n

    Args:
        vector (1xd): The array to be reversed
        n (int): chunk size

    Returns:
        Array: reversed array
    """
    
    new_vector = []
    while len(vector):
        new_vector+= vector[-n:]
        vector = vector[:-n]
    return new_vector
    

a = [2,3,4,5,6,7,8,9,10]
print(q6(a,4))

