from matplotlib import markers
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
# from matplotlib import animation
plt.style.use('seaborn-pastel')
# from tqdm import tqdm 
# import plotly.graph_objects as go 

## load and preprocess the data 

ipx_path = '/home/sidd_s/assign_data/COL774_A1/data/q1/linearX.csv'
opy_path = '/home/sidd_s/assign_data/COL774_A1/data/q1/linearY.csv'

def load_create_data(ipx_path, opy_path):
    x_data = pd.read_csv(ipx_path, header = None)
    x_data = np.array(x_data) 
    # print(x_data.ix)
    y_data = pd.read_csv(opy_path, header=None) 
    y_data = np.array(y_data)
    # print(y_data.dtype)
    # print(x_data.shape)
    # print(x_data)
    # print(y_data)
    # print(y_data.shape)
    return x_data, y_data  

def normalise_data(x_data):
    ## normalise the data 
    x_mean= np.mean(x_data,axis=0) 
    x_std=np.std(x_data , axis=0)
    # print(x_mean, x_std)  
    x_norm_data = (x_data - x_mean) / x_std 
    x_ones = np.ones(x_norm_data.shape[0]) 
    x_ones= np.expand_dims(x_ones, axis=1) 
    # print(x_ones.shape)
    # print(x_norm_data.shape)
    x_norm_data = np.hstack((x_ones, x_norm_data))
    # print(x_norm_data)    
    # print(x_norm_data.shape) 
    return x_norm_data

x_data, y_data = load_create_data(ipx_path, opy_path)
x_norm_data = normalise_data(x_data)

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

## loss function 
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

## Training the model 
def model_fit(epochs, lr, x_data, y_data, theta):
    counter = 0 
    future_counter = 1
    run_past_avg_cost = 0.0  
    run_future_avg_cost = 0.0 
    avg_diff_cost = []
    # theta_lst = [] 
    cost_lst = []
    theta0_lst = []
    theta1_lst = []
    # eps = 1.2e-6  #  (for lr 0.1))
    eps = 1.2e-1 #(lr 0.001) 
    for ep in range(epochs):  
        y_pred = pred(x_data, theta) 
        J_theta = loss(y_data, y_pred) 
        theta = grad_descent(theta, x_data, y_data, y_pred, lr)  
        # theta_lst.append(theta)  
        
        cost_lst.append(J_theta)
        theta0_lst.append(theta[0].tolist()[0])  
        # print(theta0_lst) 
        theta1_lst.append(theta[1].tolist()[0])

        ## convergence criteria  
        k = 2 # hyperparam  
        if epochs > k: 
            counter = counter + 1   
            run_past_avg_cost += J_theta 
            # print(counter)
            if counter == k: 
                # print('<<<<<')
                run_past_avg_cost /= k 
                theta_present = theta  
                future_counter = 0 
            if future_counter == 0: 
                counter -= 2 
                # print('>>>>>')
                run_future_avg_cost += J_theta
                if counter == 0: 
                    # print('$$$$$$$$$$$$$$$')
                    run_future_avg_cost /= k 
                    # print(run_future_avg_cost)
                    # print(run_past_avg_cost)
                    avg_cost_diff = abs(run_past_avg_cost - run_future_avg_cost) 
                    # print(avg_cost_diff) 
                    avg_diff_cost.append(avg_cost_diff)
                    if avg_cost_diff < eps:
                        print('^^^^^^')
                        print(ep)
                        return theta_present, cost_lst, theta0_lst, theta1_lst 
                    else: 
                        future_counter = 1  

    # plt.plot(avg_diff_cost)  
    # plt.plot(cost_lst)
    # plt.savefig('try.png')
    # plt.show()
    return theta, cost_lst, theta0_lst, theta1_lst   

epochs = 12000 # hyper param
# lr = 0.1  # hyper param   94 epochs eps: 1.2e-6 k=2  
lr = 0.001 # 9185 epochs 1.2e-6 k=2
# lr = 0.025 # 367 epochs 1.2e-6 k=2
theta = np.zeros((2,1))  ## intial param  
# print(theta.shape) 
# theta = np.expand_dims(theta, axis=1)  


theta_optim, cost_lst, theta0_lst, theta1_lst  = model_fit(epochs, lr, x_norm_data, y_data, theta)  
print(theta_optim)    
# print(len(cost_lst))
# print(len(theta1_lst))
# print(len(theta0_lst))
# lr:0.1 # # [[0.99657029]
#  [0.00134013]]  
# lr: 0.025 
## [[0.99653052]
#  [0.00134008]]
# lr: 0.001 
# [[0.99651845]
#  [0.00134006]]

# def plot_2d(lst, color):
#     fig = plt.figure()
#     ax = plt.axes()
#     ax.plot(lst, color=color)
#     plt.show()
#     return 

# plot_2d(cost_lst, 'b')
# plot_2d(theta0_lst, 'r')
# print('finished it')

## part c 

class plot_3d_mesh:

    def __init__(self, cost_lst, theta0_lst, theta1_lst):
        self.figure = plt.figure()
        self.axes = plt.axes(projection='3d') 
        self.theta0_arr = np.array(theta0_lst)
        self.theta1_arr = np.array(theta1_lst)
        self.cost_arr = np.array(cost_lst) 
        self.theta0, self.theta1 = np.meshgrid(self.theta0_arr, self.theta1_arr)   
        self.cost = np.tile(self.cost_arr, (self.cost_arr.shape[0], 1))  
        # print(self.cost.shape[1])
        # print('*************')


    def initialise(self): 
        self.contour.set_data([], [])
        self.contour.set_3d_properties([])
        return self.contour,

    def animation(self,i):
        self.contour.set_data(self.theta0_arr[:i], self.theta1_arr[:i])
        self.contour.set_3d_properties(self.cost_arr[:i])
        return self.contour, 


    def plot(self): 
        # fig = go.Figure(data=[go.Surface(x=theta0_lst, y=theta1_lst, z=cost_lst)]) 
        # fig.update_layout(title='Loss variation with theta', autosize=True, xaxis_title='theta0', 
        #              yaxis_title='theta1') 
        # fig.show()  
          
        
        # plt.plot(theta0_arr, theta1_arr, cost_arr, color='black') 
        # axes.plot(theta0_arr, theta1_arr, cost_arr, color='black') 
        # axes2 = axes.twinx() 
        # line, = axes2.plot([], [], lw=3)
        # axes.plot_wireframe(theta0, theta1, cost, color='black')
        # print(cost.shape)
        # print(theta0.shape)  
        # print(theta0)
        # print(cost_arr.shape) # (94,)
        # axes.contour3D(theta0, theta1, cost) 
        self.axes.plot_surface(self.theta0, self.theta1, self.cost, rstride=1, cstride=1, cmap=plt.cm.plasma, alpha=0.8, edgecolor='none')  ## org  
        # axes.plot(theta0_arr, theta1_arr, cost_arr, color='black')   
        self.contour, = self.axes.plot([], [], [], 'bo', lw=1, linestyle='dashed', marker='o', markersize=4) 
        # print('************')
        anim = FuncAnimation(self.figure, self.animation, init_func=self.initialise,
                               frames=self.cost_arr.shape[0], interval=200)
        
        # axes.plot_surface(theta0, theta1, cost)
        # axes.plot_trisurf(theta0, theta1, cost, linewidth=0.3, antialiased=False)   ## for 1d case 
        # plt.show()
        # plt.savefig('q1_loss_surface3d.png') 
        # plt.savefig('try.png') 
        # print('yo')
        anim.save('q1_loss_curve_c.gif', writer='imagemagick') 
        print('yo')
        return

# print('show the plot>..')
# plot3d = plot_3d_mesh(cost_lst, theta0_lst, theta1_lst)
# plot3d.plot()


## part b 
# # plt.plot(x_norm_data, y_data)
# # print(x_norm_data[:,1].shape)
# # print(y_data[:,0].shape)
# plt.scatter(x_norm_data[:,1], y_data[:,0]) 

# theta_optim = np.array((0.99657029, 0.00134013))   
# theta_optim = np.expand_dims(theta_optim, axis=1)
# y_pred = np.dot(x_norm_data, theta_optim) 
# # print(y_pred.shape)
# plt.plot(x_norm_data[:,1], y_pred, 'r--') 
# # plt.axes('off')
# plt.show()  


## part d 

class plot2d_curve: 

    def __init__(self, cost_lst): 
        self.figure, self.axes = plt.subplots() 
        self.cost_arr = np.array(cost_lst) 
        # self.cost = np.tile(self.cost_arr, (self.cost_arr.shape[0], 1))  
        # print(cost_lst)
        self.iterations = np.arange(len(cost_lst))   
        # self.iterations = []
        # print(self.iterations)
        # print(self.cost_arr.shape) 
        # print(self.iterations.shape)

    def initialise(self): 
        self.contour.set_data([], [])
        return self.contour,   

    def animation(self,i):
        # x = np.arange(len(cost_lst))
        # print(self.iterations[:i])
        # print(self.cost_arr[:i])  
        # print(i)
        # self.iterations.append(i) 
        self.contour.set_data(self.iterations[:i], self.cost_arr[:i])
        self.axes.set_xlim([0,self.cost_arr.shape[0]])
        self.axes.set_ylim([0,np.max(self.cost_arr)])
        # self.contour.set_data(self.iterations, cost_lst[:i])
        return self.contour, 

    def plot(self):
        # self.contour, = self.axes.plot([], [], 'b', lw=1, linestyle='dashed', marker='x', markersize=4) 
        self.contour, = self.axes.plot([], [], 'b', lw=1, linestyle='dashed') 
        anim = FuncAnimation(self.figure, self.animation, init_func=self.initialise,
                               frames=self.cost_arr.shape[0], interval=200)
        anim.save('q1_loss_curve_e_001.gif', writer='imagemagick')      
        
        # print('yo') 
        return 
        

plot2d = plot2d_curve(cost_lst)
plot2d.plot()      
        
## part e 





 