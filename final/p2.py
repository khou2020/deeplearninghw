# Activate this cell to import Python packages + custom libraries

# import autograd functionalities
import autograd.numpy as np
from autograd import grad as compute_grad   

# import plotting library and other necessities
import matplotlib.pyplot as plt
from matplotlib import gridspec

# import general libraries
import copy
from datetime import datetime 

# import custom 
import normalizers
from my_convnet_lib import superlearn_setup as setup

#this is needed to compensate for matplotlib notebook's tendancy to blow up images when plotted inline
from matplotlib import rcParams
rcParams['figure.autolayout'] = True

import sys
sys.path.append('../../')

from timeit import default_timer as timer

# load in full MNIST dataset
datapath = 'data/MNIST_subset.csv'
data = np.loadtxt(datapath,delimiter = ',')

# import data and reshape appropriately
x = data[:-1,:]    # input
y = data[-1:,:]    # corresponding output

# contrast normalize our sample of images - by standard normalizing each one
normalizer,inverse_normalizer = normalizers.standard(x.T)
x = normalizer(x.T).T

# Step 1: import the setup module of our convnet library
mylib1 = setup.Setup(x,y)
# Step 3: define fully connected / multilayer perceptron layers
layer_sizes = [10,10,10];
name = 'multilayer_perceptron_batch_normalized'
super_type = 'classification'
activation = 'maxout'
mylib1.choose_features(name = name,layer_sizes = layer_sizes,super_type = super_type,activation = activation)

# Step 4: split data into training and testing sets
mylib1.make_train_val_split(train_portion = 0.8)

# Step 5: choose input normalization scheme
mylib1.choose_normalizer(name = 'ZCA_sphere')

# Step 6: choose cost function
mylib1.choose_cost(name = 'multiclass_softmax')

# Step 7: run optimization algo
mylib1.fit(max_its = 100, alpha_choice = 10**(0),batch_size = 500)

# Step 8: Plot training / validation histories
#mylib1.show_histories(start = 0)

# pluck out the highest validation accuracy from the run above
ind1 = np.argmax(mylib1.val_accuracy_histories[0])
best_result1 = mylib1.val_accuracy_histories[0][ind1]
print ('from this run our best validation accuracy was ' + str(np.round(best_result1*100,2)) + '% at step ' + str(ind1))

# number of 3x3 convolutional kernels to learn (set by you)
num_kernels = 8

# convolution stride (set by you)
conv_stride = 2

# Step 1: import the setup module of our convnet library
mylib2 = setup.Setup(x,y)

# Step 2: define convolution layer
kernel_sizes = [num_kernels,3,3]
pool_stride = 2
mylib2.choose_convolutions(kernel_sizes = kernel_sizes,conv_stride = conv_stride, pool_stride = pool_stride)

# Step 3: define fully connected / multilayer perceptron layers
layer_sizes = [10,10,10];
name = 'multilayer_perceptron_batch_normalized'
super_type = 'classification'
activation = 'maxout'
mylib2.choose_features(name = name,layer_sizes = layer_sizes,super_type = super_type,activation = activation,scale = 0.1)

# Step 4: split data into training and testing sets
mylib2.x_train = mylib1.x_train
mylib2.y_train = mylib1.y_train
mylib2.x_val = mylib1.x_val
mylib2.y_val = mylib1.y_val

# Step 5: choose input normalization scheme
mylib2.choose_normalizer(name = 'ZCA_sphere')

# Step 6: choose cost function
mylib2.choose_cost(name = 'multiclass_softmax')

# Step 7: run optimization algo
mylib2.fit(max_its = 20, alpha_choice = 10**(0), batch_size = 500)

# Step 8: Plot training / validation histories
mylib2.show_histories(start = 0)

# pluck out the highest validation accuracy from the run above
ind2 = np.argmax(mylib2.val_accuracy_histories[0])
best_result2 = mylib2.val_accuracy_histories[0][ind2]
print ('from this run our best validation accuracy was ' + str(np.round(best_result2*100,2)) + '% at step ' + str(ind2))