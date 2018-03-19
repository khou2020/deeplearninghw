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

# load in X_A: input data for task A
X_A = np.loadtxt('data/X_A.txt', delimiter=',')

# load in X_B: input data for task B
X_B = np.loadtxt('data/X_B.txt', delimiter=',')

# load in y1: one of the label vectors, we don't know what task it belongs to!  
y1 = np.reshape(np.loadtxt('data/y1.txt', delimiter=','), (-1,1)).T

# load in y2: the other label vector! we don't know what task it belongs to! 
y2 = np.reshape(np.loadtxt('data/y2.txt', delimiter=','), (-1,1)).T

# contrast normalize our sample of images - by standard normalizing each one
normalizer,inverse_normalizer = normalizers.standard(X_A.T)
X_A = normalizer(X_A.T).T
normalizer,inverse_normalizer = normalizers.standard(X_B.T)
X_B = normalizer(X_B.T).T

unique, counts = np.unique(y1, return_counts=True)
print(list(zip(unique, counts)))
unique, counts = np.unique(y2, return_counts=True)
print(list(zip(unique, counts)))

# classify on a combination of them
for x, lx in zip([X_A, X_B], ['X_A', 'X_B']):
    for y, ly in zip([y1, y2], ['y1', 'y2']):
        print("Training on " + lx + " with " + ly)

        mylib1 = setup.Setup(x,y)
        layer_sizes = [10];
        name = 'multilayer_perceptron_batch_normalized'
        super_type = 'classification'
        activation = 'relu'
        mylib1.choose_features(name = name,layer_sizes = layer_sizes,super_type = super_type,activation = activation)

        # Step 4: split data into training and testing sets
        mylib1.make_train_val_split(train_portion = 0.8)

        # Step 5: choose input normalization scheme
        mylib1.choose_normalizer(name = 'standard')

        # Step 6: choose cost function
        mylib1.choose_cost(name = 'softmax')

        # Step 7: run optimization algo
        mylib1.fit(max_its = 10, alpha_choice = 10**(0), batch_size = 100)

        # Step 8: Plot training / validation histories
        mylib1.show_histories(start = 0)

        # pluck out the highest validation accuracy from the run above
        ind1 = np.argmax(mylib1.val_accuracy_histories[0])
        best_result1 = mylib1.val_accuracy_histories[0][ind1]
        print ('from this run our best validation accuracy was ' + str(np.round(best_result1*100,2)) + '% at step ' + str(ind1))