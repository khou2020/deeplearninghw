# import necessary library
import autograd.numpy as np   
from autograd import value_and_grad 
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.autolayout'] = True

data = np.loadtxt('mnist_test_contrast_normalized.csv',delimiter = ',')
x = data[:,:-1].T
y = data[:,-1:]

# Normalize data
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   

    # create standard normalizer function based on input data statistics
    normalizer = lambda data: (data - x_means)/x_stds
    
    # return normalizer and inverse_normalizer
    return normalizer, x_means, x_stds
# return normalization functions based on input x
normalizer, mean, stddev = standard_normalizer(x)
# normalize input by subtracting off mean and dividing by standard deviation
x = normalizer(x)

# compute C linear combinations of input point, one per classifier
def model(x,w):
    # tack a 1 onto the top of each input point all at once
    o = np.ones((1,np.shape(x)[1]))
    x = np.vstack((o,x))
    
    # compute linear combination and return
    a = np.dot(x.T,w)
    return a

# multiclass softmaax regularized by the summed length of all normal vectors
def multiclass_softmax(w):        
    lam = 0

    # pre-compute predictions on all points
    all_evals = model(x,w)
    
    # compute softmax across data points
    a = np.log(np.sum(np.exp(all_evals),axis = 1)) 
    
    # compute cost in compact form using numpy broadcasting
    b = all_evals[np.arange(len(y)),y.astype(int).flatten()]
    cost = np.sum(a - b)
    
    # add regularizer
    cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2
    
    # return average
    return cost/float(len(y))

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

# multiclass counting cost
def multiclass_counting_cost(w):                
    # pre-compute predictions on all points
    all_evals = model(x,w)

    # compute predictions of each input point
    y_predict = (np.argmax(all_evals,axis = 1))[:,np.newaxis]

    # compare predicted label to actual label
    count = np.sum(np.abs(np.sign(y - y_predict)))

    # return number of misclassifications
    return count

# Search for best step size
w = 0.1*np.random.randn(x.shape[0] + 1,10)
ba = 0
bc = 10000000000
itr = range(101)
alpha = list(range(2, -5, -1))
for a in alpha:
    wh, ch = gradient_descent(multiclass_softmax, 10 ** a, 100, w)
    if (bc > ch[-1]):
        ba = a
        bc = ch[-1]
    plt.plot(itr, ch)
plt.legend([str(t) for t in alpha],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


    cl = [multiclass_counting_cost(t) for t in wh]
    plt.figure(1)
    plt.plot(itr, ch)
    plt.figure(2)
    plt.plot(itr, cl)
plt.figure(1)
plt.legend([str(t) for t in alpha],loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure(2)
plt.legend([str(t) for t in alpha],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()