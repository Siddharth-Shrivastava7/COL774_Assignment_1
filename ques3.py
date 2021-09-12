import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

## load data 
ipx_path = '/home/sidd_s/assign_data/COL774_A1/data/q3/logisticX.csv' 
opy_path = '/home/sidd_s/assign_data/COL774_A1/data/q3/logisticY.csv'

def load_create_data(ipx_path, opy_path):
    x_data = pd.read_csv(ipx_path, header = None)
    x_data = np.array(x_data) 
    y_data = pd.read_csv(opy_path, header=None) 
    y_data = np.array(y_data)
    return x_data, y_data   

# x_data, y_data = load_create_data(ipx_path, opy_path)
# print(x_data.shape) # (100, 2) 
# print(y_data.shape) # (100, 1) 
# print(x_data)

def normalise_data(x_data):
    x_mean= np.mean(x_data,axis=0) 
    x_std=np.std(x_data , axis=0)
    x_norm_data = (x_data - x_mean) / x_std 
    x_ones = np.ones(x_norm_data.shape[0]) 
    x_ones= np.expand_dims(x_ones, axis=1) 
    x_norm_data = np.hstack((x_ones, x_norm_data))
    return x_norm_data

# x_norm_data = normalise_data(x_data)
# print(x_norm_data.shape) # (100, 3) 
# print(x_norm_data)

# print(x_norm_data.shape[1])
# theta = np.zeros((x_norm_data.shape[1],1)) 

def pred_sigmoid(x_data, theta):
    # print(x_data.dtype)  # float64 
    # print(theta.dtype)   # float64
    # print(theta.shape)  # (2,1)
    # print(x_data.shape) # (100, 2)
    y_pred = np.dot(x_data, theta) 
    # print(y_pred)
    # print(y_pred.shape) # (100, 1)
    y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))
    return y_pred_sigmoid
    #  print(y_pred.shape)  

# y_pred_sigmoid = pred_sigmoid(x_norm_data, theta) 
# print(y_pred_sigmoid.shape) # (100, 1) 
# print(y_pred_sigmoid) 

def grad_logistic(x,y, y_pred):
    grad = -1* np.dot(np.transpose(x), np.sum((y - y_pred), axis=1)) / y.shape[0]   
    grad = np.expand_dims(grad, axis=1) 
    return grad 

def hessian_logistic(x,y_pred):
    hessian = np.linalg.multi_dot([np.transpose(x), y_pred, np.transpose(1-y_pred), x]) / x.shape[0]
    return hessian

# hessian = hessian_logistic(x_norm_data, y_pred_sigmoid)
# print(np.linalg.eig(hessian))

# print(hessian.shape) 
# print(hessian)  

def newton_model_fit(x,y,iterations, theta, eps): 
    grad_lst = []
    for iter in range(iterations):
        y_pred = pred_sigmoid(x,theta) 
        grad = grad_logistic(x,y,y_pred)
        hessian = hessian_logistic(x,y_pred)
        hessian_inv = np.linalg.inv(hessian) 
        theta = theta - np.dot(hessian_inv, grad) 
        grad_lst.append(grad)
        ## covergence condition on grad 
        if grad < eps: 
            print('stop iterating')
            return grad_lst
    return grad_lst


x_data, y_data = load_create_data(ipx_path, opy_path) 
x_norm_data = normalise_data(x_data)
theta = np.zeros((x_norm_data.shape[1],1))  
# print(theta.shape) # (3, 1)

y_pred_sigmoid = pred_sigmoid(x_norm_data, theta) 
# y_pred_sigmoid = pred_sigmoid(x_data, theta) 
hessian = hessian_logistic(x_norm_data, y_pred_sigmoid)  
# print(np.linalg.eig(hessian))

hessian_inv = np.linalg.inv(hessian)
# print(hessian_inv)

iterations = 100  ##hyperparam
eps = 1e-5 ##hyperparam

# grad_lst = newton_model_fit(x_norm_data, y_data, iterations, theta, eps)
# plt.plot(grad_lst)
# plt.savefig('try.png')


