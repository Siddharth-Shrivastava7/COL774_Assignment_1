import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import argparse  
parser = argparse.ArgumentParser(description='ques_4') 

parser.add_argument('--option', type=str, default='a',
                    help='sub_question')  


def main(args):  
    ## load data 
    ipx_path = '/home/sidd_s/assign_data/COL774_A1/data/q3/logisticX.csv' 
    opy_path = '/home/sidd_s/assign_data/COL774_A1/data/q3/logisticY.csv'

    x_data, y_data = load_create_data(ipx_path, opy_path) 
    x_norm_data = normalise_data(x_data)
    theta = np.zeros((x_norm_data.shape[1],1))  ## intialise 
    iterations = 1000  ##hyperparam
    eps = 1e-16 ##hyperparam 

    if args.option == 'a': 
        theta = newton_model_fit(x_norm_data, y_data, iterations, theta, eps) 

    if args.option == 'b': 
        theta = newton_model_fit(x_norm_data, y_data, iterations, theta, eps) 
        plot_data_label_logistic(x_norm_data, theta, y_data)

    return 


def load_create_data(ipx_path, opy_path):
    x_data = pd.read_csv(ipx_path, header = None)
    x_data = np.array(x_data) 
    y_data = pd.read_csv(opy_path, header=None) 
    y_data = np.array(y_data)
    return x_data, y_data   


def normalise_data(x_data):
    x_mean= np.mean(x_data,axis=0) 
    x_std=np.std(x_data , axis=0)
    x_norm_data = (x_data - x_mean) / x_std 
    x_ones = np.ones(x_norm_data.shape[0]) 
    x_ones= np.expand_dims(x_ones, axis=1) 
    x_norm_data = np.hstack((x_ones, x_norm_data))
    return x_norm_data


def pred_sigmoid(x_data, theta):
    y_pred = np.dot(x_data, theta) 
    y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))
    return y_pred_sigmoid


def grad_logistic(x,y, y_pred):
    grad = -1* np.dot(np.transpose(x), np.sum((y - y_pred), axis=1)) / y.shape[0]   
    grad = np.expand_dims(grad, axis=1) 
    return grad 

def hessian_logistic(x,y_pred):
    diagonal = np.diag(np.diag(np.dot(y_pred, np.transpose(1-y_pred))))
    hessian = np.linalg.multi_dot([np.transpose(x), diagonal, x])  / x.shape[0]
    return hessian


def newton_model_fit(x,y,iterations, theta, eps): 
    for iter in range(iterations):
        # print(iter)
        y_pred = pred_sigmoid(x,theta) 
        grad = grad_logistic(x,y,y_pred)
        hessian = hessian_logistic(x,y_pred)
        hessian_inv = np.linalg.inv(hessian) 
        theta = theta - np.dot(hessian_inv, grad) 

        ## covergence condition on grad 
        if (abs(grad[0]) < eps) and (abs(grad[1]) < eps) and (abs(grad[2]) < eps): 
            print(iter)
            print('stop iterating')
            return theta 
    return theta

### part b

def plot_data_label_logistic(x_norm_data, theta, y_data):
    index_0 = np.argwhere(y_data==0)    
    index_1 = np.argwhere(y_data==1)   
    plt.scatter(x_norm_data[index_0[:,0], 1], x_norm_data[index_0[:,0], 2], marker='o', label = '0')
    plt.scatter(x_norm_data[index_1[:,0], 1], x_norm_data[index_1[:,0], 2], marker='x', label = '1') 

    ## line obtain by the logistic regression
    decision_boundary = theta[0] + theta[1] * x_norm_data[:,1]
    plt.plot(x_norm_data[:,1], decision_boundary, 'black', label = 'newton_optim') 
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.savefig('ques3_b.png') 
    return 


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
