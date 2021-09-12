import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd 
from mpl_toolkits import mplot3d
import argparse 


parser = argparse.ArgumentParser(description='ques_2') 

parser.add_argument('--option', type=str, default='a',
                    help='sub_question') 
parser.add_argument('--batchsize', type=int, default=1, help='batch sizes: [1, 100, 10000, 1000000]') 
parser.add_argument('--numepochs', type=int, default=100, help='number of epochs: [1, 7, 215, 39747] acc to increase in batch sizes')
parser.add_argument('--model', type=int, default=0, help='models for corresponding batch sizes: (0,1,2,3)')
parser.add_argument('--thetamove', action='store_true', default=False,
                    help='use sagnet')

def main(args):

    theta_ideal = np.array((3,1,2))  ## ideal param
    theta_ideal = np.expand_dims(theta_ideal, axis=1)
    samples = 1e6  
    # r = [1, 100, 10000, 1000000]   ## batch_sizes  
    eta_lr = 0.001 
    theta_param = np.zeros((3,1)) ## intialise 
    num_epochs = args.numepochs

    if args.option == 'a': 
        y_data, x_data = generate_data_points(theta_ideal, samples)
    
    elif args.option == 'b':
        y_data, x_data = generate_data_points(theta_ideal, samples) 
        x_data_shuffled, y_data_shuffled = shuffle_data(x_data, y_data, samples) 
        theta_optim = model_fit(x_data_shuffled, y_data_shuffled, theta_param, eta_lr, num_epochs, args)
        
    elif args.option == 'c':
        start_time = time.time()
        y_data, x_data = generate_data_points(theta_ideal, samples) 
        x_data_shuffled, y_data_shuffled = shuffle_data(x_data, y_data, samples) 
        theta_optim = model_fit(x_data_shuffled, y_data_shuffled, theta_param, eta_lr, num_epochs, args) 
        print(abs(theta_optim - theta_ideal))
        test_path = '/home/sidd_s/assign_data/COL774_A1/data/q2/q2test.csv'  
        print(time.time() - start_time)
        x_test, y_test = test_load(test_path) 
        mse, mse_ideal, rel_error = test_error(x_test, y_test, args)
        print(mse, mse_ideal, rel_error)
    
    elif args.option == 'd': 
        y_data, x_data = generate_data_points(theta_ideal, samples) 
        x_data_shuffled, y_data_shuffled = shuffle_data(x_data, y_data, samples) 
        theta_optim = model_fit(x_data_shuffled, y_data_shuffled, theta_param, eta_lr, num_epochs, args)

    return 

def generate_data_points(theta, samples):
    mu1, sigma1 = 3, np.sqrt(4) 
    x1 = np.random.RandomState(seed=7).normal(mu1, sigma1, int(samples))  
    x1 = np.expand_dims(x1, axis=1) 

    mu2, sigma2 = -1, np.sqrt(4) 
    x2 = np.random.RandomState(seed=16).normal(mu2, sigma2, int(samples))  
    x2 = np.expand_dims(x2, axis=1) 

    x0 = np.ones(int(samples)) 
    x0 = np.expand_dims(x0, axis=1) 
    x_data = np.hstack((x0,x1,x2))  

    mu_eps, sigma_eps = 0, np.sqrt(2)  
    epsilon = np.random.RandomState(seed=9).normal(mu_eps, sigma_eps, int(samples))  
    epsilon = np.expand_dims(epsilon, axis=1) 

    y_data = np.dot(x_data, theta) + epsilon  
    return y_data, x_data 


def shuffle_data(x_data, y_data, samples): 
    shuffle_indices = np.random.RandomState(seed=50).permutation(int(samples))  
    x_data_shuffled = x_data[shuffle_indices, :] 
    y_data_shuffled = y_data[shuffle_indices]   
    return x_data_shuffled, y_data_shuffled


def loss(y, y_pred): 
    J_theta = np.sum((y - y_pred)**2) / (2*y.shape[0]) 
    return J_theta


def grad_descent(theta,x, y, y_pred,lr):
    grad = -1* np.dot(np.transpose(x), np.sum((y - y_pred), axis=1)) / y.shape[0]    
    grad = np.expand_dims(grad, axis=1)
    theta -= lr*grad 
    return theta 


def pred(x_data, theta):
    y_pred = np.dot(x_data, theta) 
    return y_pred


def model_fit(x, y, theta, lr, num_epochs, args): 
    counter = 0 
    future_counter = 1
    run_past_avg_cost = 0.0  
    run_future_avg_cost = 0.0 
    theta0_lst = []
    theta1_lst = [] 
    theta2_lst = [] 
    num_iterations = y.shape[0] // args.batchsize
    avg_diff_cost = []
    for epoch in range(num_epochs):
        print(epoch)
        for iter in range(num_iterations):   
            x_batch = x[iter*args.batchsize : (iter+1) *args.batchsize, :] ## round robin
            y_batch = y[iter*args.batchsize : (iter+1) *args.batchsize]  ## round robin

            y_pred_batch = pred(x_batch, theta)
            
            J_theta = loss(y_batch, y_pred_batch)

            theta = grad_descent(theta, x_batch, y_batch, y_pred_batch, lr)    
            if args.thetamove:        
                theta0_lst.append(np.sum(theta[0]))
                theta1_lst.append(np.sum(theta[1]))
                theta2_lst.append(np.sum(theta[2]))
            
            ## convergence criteria  
            # k = 10 # hyperparam 
            # eps = 9 # hyperparam  
            # if num_iterations > k: 
            #     counter = counter + 1   
            #     run_past_avg_cost += J_theta 
            #     if counter == k: 
            #         # print('<<<<<')
            #         run_past_avg_cost /= k 
            #         theta_present = theta  
            #         future_counter = 0 
            #     if future_counter == 0: 
            #         counter -= 2 
            #         # print('>>>>>')
            #         run_future_avg_cost += J_theta
            #         if counter == 0: 
            #             # print('$$$$$$$$$$$$$$$')
            #             run_future_avg_cost /= k 
            #             avg_cost_diff = abs(run_past_avg_cost - run_future_avg_cost) 
            #             # print(avg_cost_diff)
            #             avg_diff_cost.append(avg_cost_diff)
            #             if avg_cost_diff < eps:
            #                 print('^^^^^^')
            #                 print(iter)
            #                 print(theta_present)
            #                 return theta_present 
            #             else: 
            #                 future_counter = 1      

        # print(theta)   
    
    if args.thetamove:
        theta0_arr = np.array(theta0_lst) 
        theta1_arr = np.array(theta1_lst)  
        theta2_arr = np.array(theta2_lst) 
        fig = plt.figure()   
        ax = plt.axes(projection='3d')  
        ax.plot3D(theta0_arr,theta1_arr, theta2_arr, 'blue', label = 'batchsize: 1M') 
        # ax.scatter3D(theta0_arr,theta1_arr, theta2_arr, marker='x') 

        # if args.model == 0: 
        #     theta = np.array((3.01997477, 0.99947612, 1.97689723)) 
        # elif args.model == 1: 
        #     theta = np.array((3.00062325, 0.99954274, 1.9990125))
        # elif args.model == 2:
        #     theta = np.array((2.99018629, 1.00150609, 1.99897263))
        # elif args.model == 3: 
        #     theta = np.array((2.99808766, 0.9996646, 1.99958892))

        ax.set_xlabel('$theta0$')
        ax.set_ylabel('$theta1$')
        ax.set_zlabel('$theta2$')
        ax.legend()
        plt.savefig('theta_move_1M.png')
        
    return theta 


 
def test_load(test_path):
    df = pd.read_csv(test_path)  
    x1 = np.array(df['X_1']) 
    x1 = np.expand_dims(x1, axis =1) 
    x2 = np.array(df['X_2']) 
    x2 = np.expand_dims(x2, axis =1)  
    x0 = np.ones(x1.shape[0]) 
    x0 = np.expand_dims(x0, axis=1)
    x_test = np.hstack((x0, x1, x2))
    y_test = np.array(df['Y']) 
    y_test = np.expand_dims(y_test, axis = 1) 
    return x_test, y_test


def test_error(x_test, y_test, args): 
    print('testing')
    if args.model == 0: 
        theta = np.array((3.01997477, 0.99947612, 1.97689723)) 
    elif args.model == 1: 
        theta = np.array((3.00062325, 0.99954274, 1.9990125))
    elif args.model == 2:
        theta = np.array((2.99018629, 1.00150609, 1.99897263))
    elif args.model == 3: 
        theta = np.array((2.99808766, 0.9996646, 1.99958892))
    
    theta = np.expand_dims(theta, axis=1)
    y_pred = pred(x_test, theta)  
    mse = loss(y_test, y_pred)

    theta_ideal = np.array((3,1,2))
    theta_ideal = np.expand_dims(theta_ideal, axis=1) 
    y_pred_ideal = pred(x_test, theta_ideal)
    mse_ideal = loss(y_test, y_pred_ideal) 

    rel_error = abs(mse_ideal - mse) 
    return mse, mse_ideal, rel_error


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
