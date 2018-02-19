# import custom library
from deeplearning_library_v1 import superlearn_setup
from deeplearning_library_v1 import unsuperlearn_setup

# define path to datasets
datapath = 'datasets/'

# import autograd functionality to bulid function's properly for optimizers
import autograd.numpy as np

# plotting utilities
import matplotlib.pyplot as plt
from matplotlib import gridspec

# this is needed to compensate for %matplotlib notebook's tendancy to blow up images when plotted inline
from matplotlib import rcParams
rcParams['figure.autolayout'] = True

data = np.loadtxt(datapath + 'mnist_test_contrast_normalized.csv', delimiter = ',')
x = data[:,:-1].T
y = data[:,-1:].T

demo = superlearn_setup.Setup(x,y)

# choose features
demo.choose_features(name = 'multilayer_perceptron',layer_sizes = [784, 30, 20, 10, 10], activation = 'relu')

# choose normalizer
demo.choose_normalizer(name = 'standard')

# choose cost
demo.choose_cost(name = 'multiclass_softmax')

demo.fit(max_its = 500, alpha_choice = 10**(-1), version='normalized')

demo.show_histories(start = 10, labels=['normalized run'])

demo.fit(max_its = 500, alpha_choice = 10**(-1), version='normalized', beta_choice = 0.2)
demo.fit(max_its = 500, alpha_choice = 10**(-1), version='normalized', beta_choice = 0.9)

demo.show_histories(start = 10, labels=['0', '0.2', '0.9'])