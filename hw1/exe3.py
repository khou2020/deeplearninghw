# import necessary library for this exercise
import autograd.numpy as np   
from autograd import value_and_grad 
import matplotlib.pyplot as plt

# data input
csvname = '2d_linregress_data.csv'
data = np.loadtxt(csvname,delimiter = ',')

# get input and output of dataset
x = data[:,:-1].T
y = data[:,-1:] 

# scatter plot the input data
plt.figure()
plt.scatter(x,y,color = 'k',edgecolor = 'w')
plt.show()

# Gradient Descent Code
# using an automatic differentiator - like the one imported via the statement below - makes coding up gradient descent a breeze
from autograd import value_and_grad 

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w):
    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(w)
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval
            
    # collect final weights
    weight_history.append(w)
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g(w))  
    return weight_history,cost_history

# Least Square Code

# compute linear combination of input point
def model(x,w):    
    # stack a 1 onto the top of each input point all at once
    o = np.ones((1,np.shape(x)[1]))
    x = np.vstack((o,x))
    
    # compute linear combination and return
    a = np.dot(x.T,w)
    return a

# an implementation of the least squares cost function for linear regression
def least_squares(w):    
    # compute the least squares cost
    cost = np.sum((model(x,w) - y)**2)
    return cost/float(len(y))

# Run GD
w = 0.1*np.random.randn(2,1)
h1, c1 = gradient_descent(least_squares, 0.1, 1000, w)
h2, c2 = gradient_descent(least_squares, 0.01, 1000, w)
h3, c3 = gradient_descent(least_squares, 0.001, 1000, w)
h4, c4 = gradient_descent(least_squares, 0.0001, 1000, w)
h5, c5 = gradient_descent(least_squares, 0.00001, 1000, w)

# Plot cost function history plot
import matplotlib.pyplot as plt
itr = range(len(c1))
plt.plot(itr, c1)
plt.plot(itr, c2)
plt.plot(itr, c3)
plt.plot(itr, c4)
plt.plot(itr, c5)
plt.legend(['-1', '-2', '-3', '-4', '-5'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# scatter plot the input data
x1 = np.reshape(np.array(np.min(x)), (1, 1))
x2 = np.reshape(np.array(np.max(x)), (1, 1))
y1 = model(x1, h1[-1])
y2 = model(x2, h1[-1])
plt.figure()
plt.plot([x1[0, 0], x2[0, 0]], [y1[0, 0], y2[0, 0]], 'k-')
plt.scatter(x,y,color = 'k',edgecolor = 'w')
plt.plot()
plt.show()