# import necessary library
import autograd.numpy as np   
from autograd import value_and_grad 
import matplotlib.pyplot as plt

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

# create the input function
g = lambda w: 1/float(50)*(w**4 + w**2 + 10*w)   

# RUN GRADIENT DESCENT TO MINIMIZE THIS FUNCTION
h1, c1 = gradient_descent(g, 1, 1000, 2.0)
h2, c2 = gradient_descent(g, 0.1, 1000, 2.0)
h3, c3 = gradient_descent(g, 0.01, 1000, 2.0)

# COST FUNCTION HISTORY PLOTTER GOES HERE
import matplotlib.pyplot as plt
x = range(len(c1))
plt.plot(x, c1)
plt.plot(x, c2)
plt.plot(x, c3)
plt.legend(['1','0.1', '0.01'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()