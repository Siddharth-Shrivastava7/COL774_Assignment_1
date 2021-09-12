import numpy as np
import matplotlib.pyplot as plt
import argparse  

parser = argparse.ArgumentParser(description='ques_4') 


parser.add_argument('--option', type=str, default='a',
                    help='sub_question')  

def main(args): 
    ## load the data 
    ipx_path = '/home/sidd_s/assign_data/COL774_A1/data/q4/q4x.dat'
    opy_path = '/home/sidd_s/assign_data/COL774_A1/data/q4/q4y.dat'
    x_data = np.transpose(np.loadtxt(ipx_path, unpack=True))
    y_data = np.array([0 if i.strip().split() == ['Alaska'] else 1 for i in open(opy_path).readlines()])
    y_data = y_data[...,None] # expand dims 
    x_norm_data = normalise_data(x_data)    
    index_0 = np.argwhere(y_data==0) 
    index_1 = np.argwhere(y_data==1)

    
    if args.option == 'a' or args.option == 'd': 
        mu0, mu1, sigma_shared, sigma_0, sigma_1 = lda(x_norm_data, index_0, index_1) 
        # print(mu0,mu1, sigma_shared)  
        # print("**********")
        # print(sigma_0, sigma_1)

    elif args.option == 'b': 
        plot_data_label(x_norm_data, index_0, index_1)
    
    elif args.option == 'c':
        mu0, mu1, sigma_shared, sigma_0, sigma_1 = lda(x_norm_data, index_0, index_1) 
        plot_data_label_lda(x_norm_data, index_0, index_1, sigma_shared, mu0, mu1) 
    
    elif args.option == 'e': 
        mu0, mu1, sigma_shared, sigma_0, sigma_1 = lda(x_norm_data, index_0, index_1)  
        plot_data_label_lda_qda(x_norm_data, index_0, index_1, mu0, mu1, sigma_0, sigma_1, sigma_shared)

    return 



def normalise_data(x_data):
    ## normalise the data 
    x_mean= np.mean(x_data,axis=0) 
    x_std=np.std(x_data , axis=0)
    # print(x_mean, x_std)  
    x_norm_data = (x_data - x_mean) / x_std  
    return x_norm_data


def lda(x_norm_data, index_0, index_1): 
     
    mu0 = np.sum(x_norm_data[index_0[:,0], :], axis=0) / (x_norm_data[index_0[:,0], :].shape[0])  
    mu0 = mu0.reshape((mu0.shape[0], 1)) 
    # print(mu0)  # [-0.75529433  0.68509431] 

    mu1 = np.sum(x_norm_data[index_1[:,0], :], axis=0) / (x_norm_data[index_1[:,0], :].shape[0]) 
    mu1 = mu1.reshape((mu1.shape[0], 1)) 
    # print(mu1) # [ 0.75529433 -0.68509431] 

    ## shared covariance matrix calc (by pool variances)
    x_norm_0 = x_norm_data[index_0[:,0], :] - np.transpose(mu0)  
    sigma_0 = np.dot(np.transpose(x_norm_0), x_norm_0) / x_norm_0.shape[0]
    # print(sigma_0)  
    # [[ 0.38158978 -0.15486516]
    #  [-0.15486516  0.64773717]]

    x_norm_1 = x_norm_data[index_1[:,0], :] - np.transpose(mu1)  
    sigma_1 = np.dot(np.transpose(x_norm_1), x_norm_1) / x_norm_1.shape[0]
    # print(sigma_1) 
    # [[0.47747117 0.1099206 ]
    #  [0.1099206  0.41355441]]

    ## pooling covarianc matrices 
    sigma_shared = ((x_norm_0.shape[0] - 1) * sigma_0 + (x_norm_1.shape[0] - 1)* sigma_1) / ((x_norm_0.shape[0] - 1) + (x_norm_1.shape[0] - 1)) # unbiased cov mat estimator
    # sigma_shared = ((x_norm_0.shape[0]) * sigma_0 + (x_norm_1.shape[0])* sigma_1) / ((x_norm_0.shape[0]) + (x_norm_1.shape[0])) # same here .. as compare to above(not always) 
    # [[ 0.42953048 -0.02247228]
    # [-0.02247228  0.53064579]]

    return mu0, mu1, sigma_shared, sigma_0, sigma_1


def plot_data_label(x_norm_data, index_0, index_1):
    ### part b 
    ## data 
    plt.scatter(x_norm_data[index_0[:,0], 0], x_norm_data[index_0[:,0], 1], marker='o', label='Alaska')
    plt.scatter(x_norm_data[index_1[:,0], 0], x_norm_data[index_1[:,0], 1], marker='x', label= 'Canada')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.savefig('ques4_b.png') 


def plot_data_label_lda(x_norm_data, index_0, index_1, sigma_shared, mu0, mu1): 
    plt.scatter(x_norm_data[index_0[:,0], 0], x_norm_data[index_0[:,0], 1], marker='o', label='Alaska')
    plt.scatter(x_norm_data[index_1[:,0], 0], x_norm_data[index_1[:,0], 1], marker='x', label= 'Canada')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    x_norm_0 = x_norm_data[index_0[:,0], :] - np.transpose(mu0)  
    x_norm_1 = x_norm_data[index_1[:,0], :] - np.transpose(mu1)   
    
    sigma_shared_inverse = np.linalg.inv(sigma_shared) 
    phi = x_norm_1.shape[0] / (x_norm_1.shape[0] + x_norm_0.shape[0])
    startx = np.min(x_norm_data[:,0]) 
    endx = np.max(x_norm_data[:,0])

    matrix = np.dot(np.transpose(mu1-mu0), sigma_shared_inverse) 
    constant = np.log((1-phi)/phi) 
    scalar = (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + constant

    ## div is done for normalisation in 2nd input feature space 
    starty = np.array(-(matrix[0][0]*startx - scalar)/matrix[0][1]).item()
    endy = np.array(-(matrix[0][0]*endx - scalar)/matrix[0][1]).item()  

    plotxx = [startx, endx] 
    plotyy = [starty, endy]
    plt.plot(plotxx, plotyy, '--r', label='LDA')  ## ploting using two point coordinates 
    plt.legend() 
    plt.savefig('ques4_c.png')


def plot_data_label_lda_qda(x_norm_data, index_0, index_1, mu0, mu1, sigma_0, sigma_1, sigma_shared): 

    plt.scatter(x_norm_data[index_0[:,0], 0], x_norm_data[index_0[:,0], 1], marker='o', label='Alaska')
    plt.scatter(x_norm_data[index_1[:,0], 0], x_norm_data[index_1[:,0], 1], marker='x', label= 'Canada')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    x_norm_0 = x_norm_data[index_0[:,0], :] - np.transpose(mu0)  
    x_norm_1 = x_norm_data[index_1[:,0], :] - np.transpose(mu1)   
    
    sigma_shared_inverse = np.linalg.inv(sigma_shared) 
    phi = x_norm_1.shape[0] / (x_norm_1.shape[0] + x_norm_0.shape[0])
    startx = np.min(x_norm_data[:,0]) 
    endx = np.max(x_norm_data[:,0])

    matrix = np.dot(np.transpose(mu1-mu0), sigma_shared_inverse) 
    constant = np.log((1-phi)/phi) 
    scalar = (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + constant

    ## div is done for normalisation in 2nd input feature space 
    starty = np.array(-(matrix[0][0]*startx - scalar)/matrix[0][1]).item()
    endy = np.array(-(matrix[0][0]*endx - scalar)/matrix[0][1]).item()  

    plotxx = [startx, endx] 
    plotyy = [starty, endy]
    plt.plot(plotxx, plotyy, '--r', label='LDA')  ## ploting using two point coordinates

    ###################################qda
    sigma_1_det = np.linalg.det(sigma_1)
    sigma_0_det = np.linalg.det(sigma_0)
    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)
    
    x_norm_0 = x_norm_data[index_0[:,0], :] - np.transpose(mu0)  
    x_norm_1 = x_norm_data[index_1[:,0], :] - np.transpose(mu1) 
    
    phi = x_norm_1.shape[0] / (x_norm_1.shape[0] + x_norm_0.shape[0])
    constant_q = np.log((1-phi)*np.sqrt(sigma_1_det)/(phi*np.sqrt(sigma_0_det))) 
    coeff_lin = np.dot(np.transpose(mu1), sigma_1_inv) - np.dot(np.transpose(mu0), sigma_0_inv) 
    scalarq = (np.linalg.multi_dot([np.transpose(mu1), sigma_1_inv, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_0_inv, mu0]))/2 + constant_q
    coeff_quad = (sigma_1_inv - sigma_0_inv) / 2   

    startx = np.min(x_norm_data[:,0]) 
    endx = np.max(x_norm_data[:,0])
    num_pts = 1000
    plotxx_q = []
    plotyy_q = [] 
    plotyy1_q = [] 
    spacex = np.linspace(startx, endx, num=num_pts) 

    for xx in spacex: 
        plotxx_q.append(xx)
        quad = coeff_quad[0][0]*xx**2 
        lin = (coeff_lin[0][0] + coeff_quad[0][1] + coeff_quad[1][0])*xx 
        yy = (quad - lin + scalarq) / (coeff_lin[0][1]+coeff_quad[1][1])
        yy = yy.item()
        plotyy_q.append(yy) 

    plt.plot(plotxx_q, plotyy_q, 'black', label = 'QDA')
    plt.legend()
    plt.savefig('ques4_e.png')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



