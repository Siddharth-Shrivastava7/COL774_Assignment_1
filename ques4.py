import numpy as np
import matplotlib.pyplot as plt

## load the data 
ipx_path = '/home/sidd_s/assign_data/COL774_A1/data/q4/q4x.dat'
opy_path = '/home/sidd_s/assign_data/COL774_A1/data/q4/q4y.dat'

x_data = np.transpose(np.loadtxt(ipx_path, unpack=True))
# y_data = np.transpose(np.loadtxt(opy_path, unpack=True)) ## error since word alaska not float 
# y_data = np.loadtxt(opy_path, unpack=True) 

y_data = np.array([0 if i.strip().split() == ['Alaska'] else 1 for i in open(opy_path).readlines()])
y_data = y_data[...,None] # expand dims 

# print(x_data.shape) # (100, 2)
# print(y_data)
# print(len(y_data))  

### part a
def normalise_data(x_data):
    ## normalise the data 
    x_mean= np.mean(x_data,axis=0) 
    x_std=np.std(x_data , axis=0)
    # print(x_mean, x_std)  
    x_norm_data = (x_data - x_mean) / x_std  
    return x_norm_data

x_norm_data = normalise_data(x_data)    
# print(x_norm_data.shape) # (100, 2) 
# print(x_norm_data)
## we don't need to normalise it first ## wrong...we need too

# print(np.mean(x_norm_data, axis=0))
# print(np.std(x_norm_data, axis=0))


## same covariance assuming 

# print(y_data.shape) # (100,1)
# print(len(y_data[y_data==0])) # 50 
# print(y_data==0)
# print(y_data[y_data==0]) 
# print(np.argwhere(y_data==0)[:,0])
# print(int((y_data==0)==['true'])) 

index_0 = np.argwhere(y_data==0)   
# print(y_data.shape)

## calc for mu0  (for the class 0)
# print(x_norm_data)
# print(np.sum(x_norm_data[index_0[:,0], :]))   
# print(index_0[:,0])
# print(len(y_data[index_0]))
# print(x_norm_data[:,0]) ## 0th feature  
# print(x_norm_data[index_0[:,0],0])
# mu0 = np.sum(x_norm_data[index_0[:,0], :], axis=0) / (len(y_data[index_0])) 
# print(mu0)  # [-0.75529433  0.68509431] 
 

# mu0 = np.sum(x_data[index_0[:,0], :], axis=0) / (len(y_data[index_0]))  
mu0 = np.sum(x_norm_data[index_0[:,0], :], axis=0) / (x_norm_data[index_0[:,0], :].shape[0])  
# print(x_norm_data[index_0[:,0], :].shape) # (50, 2)
mu0 = mu0.reshape((mu0.shape[0], 1)) 
# print(mu0)  # [[-0.75529433] , [ 0.68509431]] 
# print(mu0) # [ 98.38 429.66]
# print(x_norm_data[index_0[:,0], :])
# print(np.sum(x_norm_data[index_0[:,0], :], axis=0))



# ## calc for mu1 (for the class 1)
index_1 = np.argwhere(y_data==1)
# # print(index_1)
# mu1 = np.sum(x_norm_data[index_1[:,0], :], axis=0) / (len(y_data[index_1]))  
# print(mu1) # [ 0.75529433 -0.68509431] 


# mu1 = np.sum(x_data[index_1[:,0], :], axis=0) / (len(y_data[index_1]))  
mu1 = np.sum(x_norm_data[index_1[:,0], :], axis=0) / (x_norm_data[index_1[:,0], :].shape[0]) 
mu1 = mu1.reshape((mu1.shape[0], 1)) 
# print(mu1)  # [137.46 366.62] 
# print(mu1) # [[ 0.75529433], [-0.68509431]]
# print(x_norm_data[index_1[:,0], :])
# print(np.sum(x_norm_data[index_1[:,0], :], axis=0))



# print(x_norm_data[index_1[:,0], :] == x_norm_data[index_0[:,0], :])   
# print(x_norm_data[index_1[:,0], :])
# print(np.sum(x_norm_data[index_1[:,0], :], axis=0))
# print(x_norm_data[index_0[:,0], :])
# print(np.sum(x_norm_data[index_0[:,0], :], axis=0)) 
# print(np.sum(x_data[index_0[:,0], :], axis=0))
# print(np.sum(x_data[index_1[:,0], :], axis=0))
 
## calc for sigma (shared covariance matrix) (by pool variance)

# print(mu1.shape) # (2, 1)
# print((x_data[index_1[:,0], :] - np.transpose(mu1)).shape) #  (50, 2)
# print((x_data[index_0[:,0], :] - np.transpose(mu0)))

# sigma = (np.dot((x_data[index_0[:,0], :] - np.transpose(mu0)), np.transpose((x_data[index_0[:,0], :] - np.transpose(mu0)))) + np.dot((x_data[index_1[:,0], :] - np.transpose(mu1)), np.transpose((x_data[index_1[:,0], :] - np.transpose(mu1))))) / (len(y_data))
# print(sigma.shape) 
 
## global mean calc 

# # mug = np.mean(x_data,axis=0) 
# # mug = mug.reshape((mug.shape[0],1))
# # print(mug) # [117.92 398.14]    
# # another way  
# mug2 = (mu0 * (len(y_data[index_0])) + mu1 * (len(y_data[index_1]))) / (len(y_data))  ## shared mean 
# mug2 = mug2.reshape((mug2.shape[0],1))  
# # print(mug2) # [117.92 398.14] 
# # print(mug==mug2) ## True 

## normalise data using global mean 
# # print(x_data)
# x_norm_mug = x_data - np.transpose(mug2)
# # print(x_norm_mug.shape) # (100, 2)
# # print(x_norm_mug)

# ## covariance for class 0 using global mean 
# sigma0 = np.dot(np.transpose(x_norm_mug[index_0[:,0], :]), x_norm_mug[index_0[:,0], :])
# # print(sigma0.shape) # (2, 2) 
# # print(sigma0)  

# ## covariance for class 1 using global mean 
# sigma1 = np.dot(np.transpose(x_norm_mug[index_1[:,0], :]), x_norm_mug[index_1[:,0], :])
# # print(sigma1.shape)
# # print(sigma1)

# ## shared covariance for both the classes using global mean 

# sigma_shared = (sigma0 * (len(y_data[index_0])) + sigma1 * (len(y_data[index_1]))) / (len(y_data)) 
# # print(sigma_shared)  
# # [[ 33464.68 -32132.44]
# #  [-32132.44 105838.02]]
# # print(sigma_shared.shape) # (2, 2)


# ## normalise 0 class with mu0
# x_norm_0 = x_data[index_0[:,0], :] - np.transpose(mu0) 

# ## normalise 0 class with mu0
# x_norm_1 = x_data[index_1[:,0], :] - np.transpose(mu1) 

# ## covariance for class 0 with mu0 
# sigma_0 = np.dot(np.transpose(x_norm_0), x_norm_0)

# ## covariance for class 1 with mu1
# sigma_1 = np.dot(np.transpose(x_norm_1), x_norm_1) 

# ## single shared covariance 
# sigma_shared = (sigma_0 + sigma_1) / (y_data.shape[0])

# print(sigma_shared.shape)
# sigma_shared_inverse = np.linalg.inv(sigma_shared) 
# print(sigma_shared_inverse) 

## global mean 
# [[4.21776026e-05 1.28051270e-05]
# [1.28051270e-05 1.33360391e-05]] 

# ## single mean.. 
# [[3.48620276e-03 8.30170945e-05]
# [8.30170945e-05 8.92250649e-04]]


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
# print(x_norm_0.shape) # (50, 2)
# print(sigma_shared.shape)
# print((x_norm_0.shape[0] - 1))
# print(sigma_shared)   


### part b 
## data 
# plt.scatter(x_data[index_0[:,0], 0], x_data[index_0[:,0], 1], marker='o')
# plt.scatter(x_data[index_1[:,0], 0], x_data[index_1[:,0], 1], marker='x') 
plt.scatter(x_norm_data[index_0[:,0], 0], x_norm_data[index_0[:,0], 1], marker='o')
plt.scatter(x_norm_data[index_1[:,0], 0], x_norm_data[index_1[:,0], 1], marker='x')
# plt.savefig('ques4_b.png') 


# ### part c  
sigma_shared_inverse = np.linalg.inv(sigma_shared)
# print(sigma_shared_inverse)
# sample_proportion = (len(y_data[index_0])/len(y_data[index_1])) ## class prior ratio 
phi = x_norm_1.shape[0] / (x_norm_1.shape[0] + x_norm_0.shape[0])  ## bernoulli param for class prior 
# # decision_boundary = np.transpose(np.linalg.multi_dot([np.transpose(mu1-mu0), sigma_shared_inverse, np.transpose(x_data)])) - (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + np.log(sample_proportion)  ## whichever value is grater than 0 is for the class 1 and vice versa.  

# # print(decision_boundary.shape) # (100, 1)
# # print(sample_proportion)
# # print(sigma_shared_inverse) 
# # print(decision_boundary) 

# y_pred_1 = np.transpose(np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, np.transpose(x_data)])) - (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]))/2 + np.log(sample_proportion) 
# y_pred_0 = np.transpose(np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, np.transpose(x_data)])) - (np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + np.log(sample_proportion) 
# # print(y_pred_0.shape) # (100, 1)
# # print(y_pred_1.shape) # (100, 1)
# # print(y_pred_0)
# # print(y_pred_1)

# # y_pred = np.maximum(y_pred_0, y_pred_1)  
# # plt.plot(x_data[:,0], x_data[:,1]) 

# # print(y_pred.shape) # (100, 1) 
# # print(y_pred) 

# matrix = np.transpose(np.linalg.multi_dot([np.transpose(mu1-mu0), sigma_shared_inverse, np.transpose(x_norm_data)])) 
# scalar = ((np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + np.log((1-phi)/phi))

# decision_boundary = -(np.transpose(np.linalg.multi_dot([np.transpose(mu1-mu0), sigma_shared_inverse, np.transpose(x_norm_data)])) - (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2) + np.log((1-phi)/phi)  ## correct one for normalised data 

# decision_boundary = np.transpose(np.linalg.multi_dot([np.transpose(mu1-mu0), sigma_shared_inverse, np.transpose(x_data)])) - (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + np.log((1-phi)/phi) ## not to be used
# decision_boundary = matrix - scalar
# print(decision_boundary)
# print(decision_boundary.shape) # (100, 1)     
# print(decision_boundary)  
# print(decision_boundary[index_0[:,0]]<0) 
# print(decision_boundary[index_0[:,0]])
# print(phi)  

startx = np.min(x_norm_data[:,0]) 
endx = np.max(x_norm_data[:,0])
# print(startx) # -2.5094016260593213
# print(endx)
# print(x_norm_data) 

matrix = np.dot(np.transpose(mu1-mu0), sigma_shared_inverse) 
# print(matrix.shape) # (1, 2)
constant = np.log((1-phi)/phi) 
scalar = (np.linalg.multi_dot([np.transpose(mu1), sigma_shared_inverse, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_shared_inverse, mu0]))/2 + constant

## div is done for normalisation in 2nd input feature space 
starty = np.array(-(matrix[0][0]*startx - scalar)/matrix[0][1]).item()
endy = np.array(-(matrix[0][0]*endx - scalar)/matrix[0][1]).item()  
# starty = np.max(matrix[:,0]) - scalar
# print(starty)
# print(endy) 
plotxx = [startx, endx] 
plotyy = [starty, endy]
plt.plot(plotxx, plotyy, '--r')  ## ploting using two point coordinates
# plt.savefig('try.png') 
# plt.savefig('ques4_c.png')


### part d  
# print(mu0)  
# # [[-0.75529433]
# #  [ 0.68509431]]
# print(mu1)
# # [[ 0.75529433]
# #  [-0.68509431]]
# print(sigma_0) 
# # [[ 0.38158978 -0.15486516]
# #  [-0.15486516  0.64773717]]
# print(sigma_1)
# # [[0.47747117 0.1099206 ]
# #  [0.1099206  0.41355441]] 

### part e 
sigma_1_det = np.linalg.det(sigma_1)
sigma_0_det = np.linalg.det(sigma_0)

sigma_0_inv = np.linalg.inv(sigma_0)
sigma_1_inv = np.linalg.inv(sigma_1)

# print(sigma_1_det)  
constant_q = np.log((1-phi)*np.sqrt(sigma_1_det)/(phi*np.sqrt(sigma_0_det))) 
# # x_norm_data_feat0 = x_norm_data[:,0]    
# # quad = np.transpose(np.linalg.multi_dot([x_norm_data, (sigma_1_inv - sigma_0_inv), np.transpose(x_norm_data)]))
# # print(quad.shape) 
# # print(x_norm_data_feat0.shape)   
scalarq = (np.linalg.multi_dot([np.transpose(mu1), sigma_1_inv, mu1]) - np.linalg.multi_dot([np.transpose(mu0), sigma_0_inv, mu0]))/2 + constant_q
coeff_lin = np.dot(np.transpose(mu1), sigma_1_inv) - np.dot(np.transpose(mu0), sigma_0_inv) 
# # print(coeff_lin.shape) # (1, 2) 
# # print(scalarq) 
coeff_quad = (sigma_1_inv - sigma_0_inv) / 2   
# print(coeff_quad.shape) # (2, 2)


num_pts = 1000
plotxx_q = []
plotyy_q = [] 
plotyy1_q = [] 
spacex = np.linspace(startx, endx, num=num_pts) 
# print(spacex) 
# y = coeff_quad* x**2 - coeff_lin*x + scalarq
## focusing on 0th x feature 
for xx in spacex: 
    plotxx_q.append(xx)
    # print(xx.shape)
    quad = coeff_quad[0][0]*xx**2 
    lin = (coeff_lin[0][0] + coeff_quad[0][1] + coeff_quad[1][0])*xx 
    # lin = coeff_lin[0][0]*xx
    # scalarq = scalarq + coeff_quad[1][1] + coeff_lin[0][1]
    # yy = quad - lin + scalarq   
    ## div is done for normalisation in 2nd input feature space 
    yy = (quad - lin + scalarq) / (coeff_lin[0][1]+coeff_quad[1][1])
    # aa = coeff_quad[1][1]
    # bb = (coeff_quad[0][1] + coeff_quad[1][0])*xx + coeff_lin[0][1]
    # cc = coeff_lin[0][0]*xx + coeff_quad[0][0]*xx**xx - scalarq
    # # print(yy.shape)
    # yy=(-bb+(np.sqrt(bb*bb-4.0*aa*cc)))/(2.0*aa)
    yy = yy.item()
    # yy1 = yy1.item()
    plotyy_q.append(yy) 
    # plotyy1_q.append(yy1)

# print(plotyy_q) 
# print(plotxx_q) 

# plt.plot(plotxx_q, plotyy_q, 'g')
plt.plot(plotxx_q, plotyy_q, 'black')
# plt.savefig('try.png') 
plt.savefig('ques4_e.png')
