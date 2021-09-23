from matplotlib import markers
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
import argparse 


parser = argparse.ArgumentParser(description='ques_1') 

parser.add_argument('--option', type=str, default='a',
                    help='sub_question') 
parser.add_argument('--numepochs', type=int, default=100, help='number of epochs: [94, 367, 9185] acc to decrease in lr') 
parser.add_argument('--lr', type=float, default=0.1, help='learning rate: [0.1, 0.025, 0.001]') 
 

def main(args): 
    ipx_path = '/home/sidd_s/assign_data/COL774_A1/data/q1/linearX.csv'
    opy_path = '/home/sidd_s/assign_data/COL774_A1/data/q1/linearY.csv'
    x_data, y_data = load_create_data(ipx_path, opy_path)
    x_norm_data = normalise_data(x_data)
    
    lr = args.lr 
    epochs = args.numepochs
    theta = np.zeros((2,1))  ## intial param 

    if args.option == 'a':
        theta_optim, cost_lst, theta0_lst, theta1_lst  = model_fit(epochs, lr, x_norm_data, y_data, theta) 
        # print(theta_optim)
    
    if args.option == 'b':
        plot_data_hypo(x_norm_data, y_data)
    
    if args.option == 'c': 
        theta_optim, cost_lst, theta0_lst, theta1_lst  = model_fit(epochs, lr, x_norm_data, y_data, theta) 
        plot3d = plot_3d_mesh(cost_lst, theta0_lst, theta1_lst)
        plot3d.plot()

    if args.option == 'd' or args.option == 'e':
        theta_optim, cost_lst, theta0_lst, theta1_lst  = model_fit(epochs, lr, x_norm_data, y_data, theta) 
        plot2d = plot2d_curve(cost_lst)
        plot2d.plot()   
    
    return 

## load and preprocess the data 

def load_create_data(ipx_path, opy_path):
    x_data = pd.read_csv(ipx_path, header = None)
    x_data = np.array(x_data) 
    y_data = pd.read_csv(opy_path, header=None) 
    y_data = np.array(y_data)
    return x_data, y_data  

def normalise_data(x_data):
    ## normalise the data 
    x_mean= np.mean(x_data,axis=0) 
    x_std=np.std(x_data , axis=0)
    x_norm_data = (x_data - x_mean) / x_std 
    x_ones = np.ones(x_norm_data.shape[0]) 
    x_ones= np.expand_dims(x_ones, axis=1) 
    x_norm_data = np.hstack((x_ones, x_norm_data))
    return x_norm_data


## prediction 
def pred(x_data, theta):
    y_pred = np.dot(x_data, theta) 
    return y_pred

def loss(y, y_pred): 
    J_theta = np.sum((y - y_pred)**2) / (2*y.shape[0]) 
    return J_theta

def grad_descent(theta,x, y, y_pred,lr):
    grad = -1* np.dot(np.transpose(x), np.sum((y - y_pred), axis=1)) / y.shape[0]   
    grad = np.expand_dims(grad, axis=1)
    theta -= lr*grad 
    return theta


def model_fit(epochs, lr, x_data, y_data, theta):
    counter = 0 
    future_counter = 1
    run_past_avg_cost = 0.0  
    run_future_avg_cost = 0.0 
    avg_diff_cost = []
    cost_lst = []
    theta0_lst = []
    theta1_lst = []
    eps = 1.2e-6  #  (for lr 0.1, 0.025, 0.001)) ##  convergence critera with k = 2
    for ep in range(epochs):  
        y_pred = pred(x_data, theta) 
        J_theta = loss(y_data, y_pred) 
        theta = grad_descent(theta, x_data, y_data, y_pred, lr)  
        
        cost_lst.append(J_theta)
        theta0_lst.append(theta[0].tolist()[0])  
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

    # plt.plot(avg_diff_cost)  ## for estimating eps value
    # plt.plot(cost_lst)
    # plt.savefig('try.png')
    # plt.show()
    return theta, cost_lst, theta0_lst, theta1_lst   


def plot_data_hypo(x_norm_data, y_data):

    plt.scatter(x_norm_data[:,1], y_data[:,0]) 

    theta_optim = np.array((0.99657029, 0.00134013))   
    theta_optim = np.expand_dims(theta_optim, axis=1)
    y_pred = np.dot(x_norm_data, theta_optim) 
    plt.plot(x_norm_data[:,1], y_pred, 'r-') 
    # plt.axes('off')
    plt.xlabel('normalised_input') 
    plt.ylabel('ouput_value')
    plt.savefig('ques1_b.png') 
    return 


class plot_3d_mesh:

    def __init__(self, cost_lst, theta0_lst, theta1_lst):
        self.figure = plt.figure()
        self.axes = plt.axes(projection='3d') 
        self.theta0_arr = np.array(theta0_lst)
        self.theta1_arr = np.array(theta1_lst)
        self.cost_arr = np.array(cost_lst) 
        self.theta0, self.theta1 = np.meshgrid(self.theta0_arr, self.theta1_arr)   
        self.cost = np.tile(self.cost_arr, (self.cost_arr.shape[0], 1))  
        self.axes.set_xlabel('$theta0$')
        self.axes.set_ylabel('$theta1$')
        self.axes.set_zlabel('$loss$')

    def initialise(self): 
        self.contour.set_data([], [])
        self.contour.set_3d_properties([])
        return self.contour,

    def animation(self,i):
        self.contour.set_data(self.theta0_arr[:i], self.theta1_arr[:i])
        self.contour.set_3d_properties(self.cost_arr[:i])
        return self.contour, 


    def plot(self): 
        self.axes.plot_surface(self.theta0, self.theta1, self.cost, rstride=1, cstride=1, cmap=plt.cm.plasma, alpha=0.8, edgecolor='none') 
        self.contour, = self.axes.plot([], [], [], 'bo', lw=1, linestyle='dashed', marker='o', markersize=4) 
        anim = FuncAnimation(self.figure, self.animation, init_func=self.initialise,
                               frames=self.cost_arr.shape[0], interval=200)

        anim.save('q1_loss_curve_c.gif', writer='imagemagick') 
        # print('yo')
        return


class plot2d_curve: 

    def __init__(self, cost_lst): 
        self.figure, self.axes = plt.subplots() 
        self.cost_arr = np.array(cost_lst) 
        self.iterations = np.arange(len(cost_lst)) 
        self.axes.set_xlabel('$iterations$')
        self.axes.set_ylabel('$loss$')
        

    def initialise(self): 
        self.contour.set_data([], [])
        return self.contour,   

    def animation(self,i):
        self.contour.set_data(self.iterations[:i], self.cost_arr[:i])
        self.axes.set_xlim([0,self.cost_arr.shape[0]])
        self.axes.set_ylim([0,np.max(self.cost_arr)])
        return self.contour, 

    def plot(self):
        self.contour, = self.axes.plot([], [], 'b', lw=1, linestyle='dashed', marker='x', markersize=4) 
        anim = FuncAnimation(self.figure, self.animation, init_func=self.initialise,
                               frames=self.cost_arr.shape[0], interval=200)
        anim.save('q1_loss_curve_e_001.gif', writer='imagemagick')      
        # print('yo') 
        return 
           

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




 