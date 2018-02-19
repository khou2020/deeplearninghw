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

# load data
csvname = datapath + 'noisy_sin_sample.csv'
data = np.loadtxt(csvname,delimiter = ',')
x = data[:-1,:]
y = data[-1:,:] 

# import the v1 library
demo = superlearn_setup.Setup(x,y)

# choose features
demo.choose_features(name = 'multilayer_perceptron',layer_sizes = [1,10,10,10,10,1],activation = 'tanh')

# choose normalizer
demo.choose_normalizer(name = 'standard')

# choose cost
demo.choose_cost(name = 'least_squares')

# fit an optimization
demo.fit(max_its = 100,alpha_choice = 10**(-1))

# plot cost history
demo.show_histories(start = 10,labels = ['my run'])

# load data
csvname = datapath + 'signed_projectile.csv'
data = np.loadtxt(csvname,delimiter = ',')
x = data[:-1,:]
y = data[-1:,:] 

# import the v1 library
demo = superlearn_setup.Setup(x,y)

# choose features
demo.choose_features(name = 'multilayer_perceptron',layer_sizes = [1,10,10,10,10,1],activation = 'tanh')

# choose normalizer
demo.choose_normalizer(name = 'standard')

# choose cost
demo.choose_cost(name = 'softmax')

# fit an optimization
demo.fit(max_its = 1000,alpha_choice = 10**(-1))

# plot cost history
demo.show_histories(start = 10,labels = ['run 1','run 2'])

csvname = datapath + '3_layercake_data.csv'
data = np.loadtxt(csvname,delimiter = ',')
x = data[:-1,:]
y = data[-1:,:] 

# import the v1 library
demo = superlearn_setup.Setup(x,y)

# choose features
demo.choose_features(name = 'multilayer_perceptron',layer_sizes = [2,10,10,10,10,3],activation = 'tanh')

# choose normalizer
demo.choose_normalizer(name = 'standard')

# choose cost
demo.choose_cost(name = 'multiclass_softmax')

# fit an optimization
demo.fit(max_its = 1000,alpha_choice = 10**(0))

# plot cost history
demo.show_histories(start = 10)