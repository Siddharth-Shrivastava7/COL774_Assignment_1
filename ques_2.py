import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd 
from mpl_toolkits import mplot3d
# from torch.utils.tensorboard import SummaryWriter 
# writer = SummaryWriter('log_dir')  

### part a 

start_time = time.time()   

def generate_data_points(theta, samples):
    mu1, sigma1 = 3, np.sqrt(4) # given 
    # x1 = np.random.normal(mu1, sigma1, int(samples))  
    x1 = np.random.RandomState(seed=7).normal(mu1, sigma1, int(samples))  
    x1 = np.expand_dims(x1, axis=1) 
    # print(x1.shape) # (1000000,1) 
    # print(x1)
    mu2, sigma2 = -1, np.sqrt(4) # given 
    # x2 = np.random.normal(mu2, sigma2, int(samples))  
    x2 = np.random.RandomState(seed=16).normal(mu2, sigma2, int(samples))  
    x2 = np.expand_dims(x2, axis=1) 
    x0 = np.ones(int(samples)) # given
    x0 = np.expand_dims(x0, axis=1) 
    x_data = np.hstack((x0,x1,x2))  
    # print(x_data.shape)   # (1000000, 3) 

    mu_eps, sigma_eps = 0, np.sqrt(2) # given 
    # epsilon = np.random.normal(mu_eps, sigma_eps, int(samples))  
    epsilon = np.random.RandomState(seed=9).normal(mu_eps, sigma_eps, int(samples))  
    epsilon = np.expand_dims(epsilon, axis=1) 

    y_data = np.dot(x_data, theta) + epsilon  
    # print(y_data.shape)  # (1000000,1) 
    # y_data_2 = theta[0] + theta[1] * x1 + theta[2] * x2 + epsilon   
    # print(np.all(y_data == y_data_2))  # True

    return y_data, x_data 


# def examplePlot():
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')  
#     # do some plotting
#     return fig

theta = np.array((3,1,2))   
theta = np.expand_dims(theta, axis=1)
samples = 1e6  
y_data, x_data = generate_data_points(theta, samples)   ## generating data points 
# print(y_data)

# print(y_data.shape)  
# print(x1.shape)
# print(x2.shape)

### part a  

### part b 

## shuffle the examples (initial step)

## fixed random sequence for repeatable result  
# np.random.seed(1)   ## not using .. not efficient 
# rng = np.random.RandomState(2021) 
# shuffle_indices = np.random.permutation(int(samples))  

def shuffle_data(x_data, y_data): 
    shuffle_indices = np.random.RandomState(seed=50).permutation(int(samples))  
    # print(shuffle_indices)
    # # input 
    x_data_shuffled = x_data[shuffle_indices, :] 
    # print(x_data_shuffled.shape) #(1000000, 3)   
    # x1_shuffled = x1[shuffle_indices] 
    # x2_shuffled = x2[shuffle_indices]
    # # output GT 
    y_data_shuffled = y_data[shuffle_indices]   
    # print(y_data_shuffled.shape)  # (1000000, 1)    
    # ## shuffle the examples (initial step)
    return x_data_shuffled, y_data_shuffled

x_data_shuffled, y_data_shuffled = shuffle_data(x_data, y_data)

## loss fun
def loss(y, y_pred): 
    # print(y.shape)
    # print(np.squeeze(y))
    J_theta = np.sum((y - y_pred)**2) / (2*y.shape[0]) 
    # print(J_theta)
    return J_theta

## gradient descent 
def grad_descent(theta,x, y, y_pred,lr):
    # print(y.shape) # (100,1)
    # print(y_pred.shape) # (100,1)
    grad = -1* np.dot(np.transpose(x), np.sum((y - y_pred), axis=1)) / y.shape[0]   
    # print(grad.shape)  
    grad = np.expand_dims(grad, axis=1)
    # print(grad.shape)  # (2, 1) 
    # print(grad) 
    # grad = np.array((grad))
    theta -= lr*grad 
    # print(theta)
    # print(theta.dtype)
    return theta 

## prediction 
def pred(x_data, theta):
    # print(x_data.dtype)  # float64 
    # print(theta.dtype)   # float64
    # print(theta.shape)  # (2,1)
    # print(x_data.shape) # (100, 2)
    y_pred = np.dot(x_data, theta) 
    # print(y_pred)
    # print(y_pred.shape) # (100, 1)
    return y_pred
    #  print(y_pred.shape)  

def model_fit(x, y, theta, lr, batch_size, num_epochs): 
    counter = 0 
    future_counter = 1
    run_past_avg_cost = 0.0  
    run_future_avg_cost = 0.0 
    theta0_lst = []
    theta1_lst = [] 
    theta2_lst = [] 
    # best_J = -1e6  
    num_iterations = y.shape[0] // batch_size
    cost = []
    avg_diff_cost = []
    # print(num_iterations)
    # for iter in range(num_epochs*num_iterations):
    for epoch in range(num_epochs):
        # print(epoch)
        for iter in range(num_iterations):   ## one epoch 
            # counter = counter + 1 
        #     # input 
            x_batch = x[iter*batch_size : (iter+1) * batch_size, :] ## round robin
            # print(x_batch.shape) 
        #     x1_batch = x1_shuffled[i*r : (i+1) * r] 
        #     x2_batch = x2_shuffled[i*r : (i+1) * r] 
        #     # output GT  
            y_batch = y[iter*batch_size : (iter+1) * batch_size]  ## round robin

        #     x_batch = np.vstack((x0, x1, x2))
        #     # prediction 
        #     pred_batch = theta_model_params[0] + theta_model_params[1]*x1_batch + theta_model_params[2]*x2_batch
            # y_pred_batch = np.dot(x_batch, theta)  
            y_pred_batch = pred(x_batch, theta)
        #    # cost/loss function per batch 
            
            ## mini batch &/ stochastic gradient descent   
            # J_theta = np.sum((y_batch - y_pred_batch)**2) / (2*batch_size)  
            J_theta = loss(y_batch, y_pred_batch)
            cost.append(J_theta)  
            # print(J_theta) 
            # # grad_J_wrt_theta = np.sum((y_batch - y_pred_batch)*x_batch) / (batch_size)  ## not correct dimensionally 
            # grad_J_wrt_theta = -1 * np.dot(np.transpose(x_batch),(y_batch-y_pred_batch)) / (batch_size)  
            # # print(grad_J_wrt_theta)
            # theta = theta - lr * grad_J_wrt_theta      
            theta = grad_descent(theta, x_batch, y_batch, y_pred_batch, lr)
            
            # print(theta.shape) # (3, 1)  
            # theta_lst.append(theta)  
            # print(theta[0]) 
            theta0_lst.append(np.sum(theta[0]))
            theta1_lst.append(np.sum(theta[1]))
            theta2_lst.append(np.sum(theta[2]))
            
            ## convergence criteria  
            # k = 1000 # hyperparam 
            # eps = 1e-9 # hyperparam  
            # if num_iterations > k: 
            #     counter = counter + 1   
            #     run_past_avg_cost += J_theta 
            #     # print(counter)
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
            #                 return theta_present 
            #             else: 
            #                 future_counter = 1      
            # param update  

    # graph 
    # print('*********')
    # x = np.arange(num_iterations) 
    # y = np.array(cost)
    # plt.plot(x,y) 
    plt.plot(cost)  ## for plot command only one y function value is enough  
    # print(cost)
    # print(num_iterations) 
    # plt.plot(avg_diff_cost)
    # plt.plot(np.array(cost))

    # print(theta_lst)  
    # for arr in theta_lst:  
    #     # print(arr[0])  
    #     # print('***********')  
    #     theta0_lst.append(arr[0])
    #     theta1_lst.append(arr[1])
    #     theta2_lst.append(arr[2]) 

    # print(len(theta0_lst)) 
    # print(theta_lst)
    # print(cost)
    # theta0_arr = np.array(theta0_lst) 
    # theta1_arr = np.array(theta1_lst)  
    # theta2_arr = np.array(theta2_lst) 
    # fig = plt.figure()   
    # ax = plt.axes(projection='3d')  
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_wireframe(theta0_arr, theta1_arr, theta2_arr)
    # # ax.plot_trisurf(theta0_arr, theta1_arr, theta2_arr, linewidth=0.3, antialiased=False)  
    # # ax.plot_trisurf(theta0_arr, theta1_arr, theta2_arr)  
    # ax.scatter3D(theta0_arr,theta1_arr, theta2_arr) 
    # the0_diff = np.diff(theta0_arr)
    # the1_diff = np.diff(theta1_arr)
    # the2_diff = np.diff(theta2_arr)
    # ax.quiver(theta0_arr, theta1_arr, theta2_arr, the0_diff, the1_diff, the2_diff)  
    
    # xx, yy = np.meshgrid(theta0_arr, theta1_arr) 
    # print(xx.shape)
    # print(yy.shape)
    # print(xx)
    # print(theta0_arr) 
    # print(yy)
    # theta2_arr = theta2_arr.reshape(xx) 
    # print(theta2_arr.shape)

    # ax.contour3D(theta0_arr, theta1_arr, theta2_arr, 50, cmap='binary') 
    # plt.plot(xx, yy, theta2_arr)
    # ax.quiver()
    
    plt.show()  
    # plt.savefig('3d.png')

    # plt.savefig('try.png')
    # writer.add_figure('My plot', examplePlot(), 0) 
    # writer.close()
    return theta 

r = [1, 100, 10000, 1000000]   ## batch_sizes  
eta_lr = 0.001 
# theta_param = np.array((0, 0, 0)) ## intialise   
# theta_param = np.expand_dims(theta_param, axis=1) 
theta_param = np.zeros((3,1)) ## intialise 
num_epochs = 100

theta_optim = model_fit(x_data_shuffled, y_data_shuffled, theta_param, eta_lr, r[2], num_epochs)  ## fitting model to the data

print(theta_optim)  
# print(time.time() - start_time)


## r0_model 
## fast with convergence criteria 
# [[3.03393716]
#  [1.01647259]
#  [1.95373921]]
 
# EPOCH 

# [[3.01997477]
#  [0.99947612]
#  [1.97689723]]


## r1_model 
# [[2.81723859]
#  [1.03951594]
#  [1.98562532]]


## r2_model = 
# [[0.24456829]
#  [0.91568265]
#  [0.46548425]]


# r3_model = [[0.00399512]
#  [0.01598531]
#  [0.00400578]] 


### predictions  (for test)

test_path = '/home/sidd_s/assign_data/COL774_A1/data/q2/q2test.csv'  

def test_load(test_path):
    df = pd.read_csv(test_path)  
    # print(df) 
    # print(df['X_1']) 
    x1 = np.array(df['X_1']) 
    x1 = np.expand_dims(x1, axis =1) 
    x2 = np.array(df['X_2']) 
    x2 = np.expand_dims(x2, axis =1)  
    # print(x1.shape) 
    # print(x2.shape)
    x0 = np.ones(x1.shape[0]) 
    x0 = np.expand_dims(x0, axis=1)
    # print(x0.shape)
    x_test = np.hstack((x0, x1, x2))
    y_test = np.array(df['Y']) 
    y_test = np.expand_dims(y_test, axis = 1) 
    # print(y_test.shape)
    return x_test, y_test

x_test, y_test = test_load(test_path)

def prediction(x, theta):
    ## original hypothesis  
    y_pred = np.dot(x_test, theta)
    return y_pred 

def MSE(y, y_pred):
    mse = np.sum((y - y_pred)**2) / (y.shape[0])
    return mse



# #mse for original hyopthesis   
# y_pred = prediction(x_test, theta)  
# MSE_org = MSE(y_test, y_pred)
# # print(MSE_org) # 1.965893843

# ## hypo for r0 learned model 
# # r0 epoch 
# theta_ro_ep =  np.array((3.01997477, 0.99947612, 1.97689723))
# # theta_ro_ep = np.expand_dims(theta_ro_ep, axis=1)
# theta_ro_ep = theta_ro_ep[...,np.newaxis]
# y_pred_ro_ep = np.dot(x_test, theta_ro_ep)
# # mse 
# MSE_ro_ep = np.sum((y_test - y_pred_ro_ep)**2) / (x1.shape[0])
# # print(MSE_ro_ep) # 2.0278262307193518 

# ## hypo for r0 learned model 
# # ro by convergence 
# theta_ro_cov = np.array((3.03393716 ,  1.01647259 , 1.95373921))
# theta_ro_cov = np.expand_dims(theta_ro_cov, axis=1)
# y_pred_ro_cov = np.dot(x_test, theta_ro_cov)
# #mse 
# MSE_ro_cov = np.sum((y_test - y_pred_ro_cov)**2) / (x1.shape[0]) 
# # print(MSE_ro_cov)   # 2.2303108247356773   ## no not good enough to test 

# ## hypo for r1 learned model 
# theta_r1 = np.array((2.81723859, 1.03951594, 1.98562532 )) 
# theta_r1 = np.expand_dims(theta_r1, axis=1) 
# y_pred_r1 = np.dot(x_test, theta_r1)  
# # mse 
# MSE_r1 = np.sum((y_test - y_pred_r1)**2) / (x1.shape[0]) 
# # print(MSE_r1)  # 2.159324519029886  

# ## hypo for r2 learned model 
# theta_r2 = np.array((0.24456829, 0.91568265, 0.46548425)) 
# theta_r2 = np.expand_dims(theta_r2, axis=1)  
# y_pred_r2 = np.dot(x_test, theta_r2) 
# # mse 
# MSE_r2 = np.sum((y_test - y_pred_r2)**2) / (x1.shape[0]) 
# # print(MSE_r2) # 241.67350507149743 

# ## hypo for r3 learned model 
# theta_r3 = np.array((0.00399512, 0.01598531, 0.00400578)) 
# theta_r3 = np.expand_dims(theta_r3, axis=1)  
# y_pred_r3 = np.dot(x_test, theta_r3) 
# # mse 
# MSE_r3 = np.sum((y_test - y_pred_r3)**2) / (x1.shape[0]) 
# # print(MSE_r3) # 503.089313071857

