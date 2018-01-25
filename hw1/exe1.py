from matplotlib import rcParams
import autograd.numpy as np  
import matplotlib.pyplot as plt
# import statment for gradient calculator
from autograd import grad    
from autograd import value_and_grad   

# import three-dimensional plotting library into namespace
from mpl_toolkits.mplot3d import Axes3D

rcParams['figure.autolayout'] = True
# a named Python function
def my_func(w):
    return np.tanh(w)
# how to use the 'sin' function
w_val = 1.0   # a test input for our 'sin' function
g_val = my_func(w_val)
print (g_val)
# how to use 'lambda' to create an "anonymous" function - just a pithier way of writing functions in Python
g = lambda w: np.tanh(w)
# how to use the 'sin' function
w_val = 1.0   
g_val = g(w_val)
print (g_val)

# if we input a float value into our function, it returns a float
print (type(w_val))
print (type(g_val))

# if we input a numpy array, it returns a numpy array
w_val = np.array([1.0])  
g_val = g(w_val)
print (g_val)
print (type(w_val))
print (type(g_val))

# create a sample of points to plot over 
w_vals = np.linspace(-5,5,200)

# evaluate the function over each of these values - one can use an explicit for-loop here instead of a list comprehension
g_vals = [g(v) for v in w_vals]

# plot
fig, ax = plt.subplots(1, 1, figsize=(6,3))
ax.plot(w_vals,g_vals)
plt.show()

# create the derivative/gradient function of g --> called dgdw
dgdw = grad(g)

# evaluate the gradient function at a point
w_val = 1.0
print (dgdw(1.0))

# create space over which to evaluate function and gradient
w_vals = np.linspace(-5,5,200)

# evaluate gradient over input range
g_vals = [g(v) for v in w_vals]
dg_vals = [dgdw(v) for v in w_vals]

# create figure
fig = plt.figure(figsize = (7,3))

# plot function and gradient values
plt.plot(w_vals,g_vals)
plt.plot(w_vals,dg_vals)
plt.legend(['func','derivative'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# compute the second derivative of our input function
dgdw2 = grad(dgdw)

# define set of points over which to plot function and first two derivatives
w = np.linspace(-5,5,500)

# evaluate the input function g, first derivative dgdw, and second derivative dgdw2 over the input points
g_vals = [g(v) for v in w]
dg_vals = [dgdw(v) for v in w]
dg2_vals = [dgdw2(v) for v in w]

# plot the function and derivative
fig = plt.figure(figsize = (7,3))
plt.plot(w,g_vals,linewidth=2)
plt.plot(w,dg_vals,linewidth=2)
plt.plot(w,dg2_vals,linewidth=2)
plt.legend(['$g(w)$',r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',r'$\frac{\mathrm{d}^2}{\mathrm{d}w^2}g(w)$'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 13)
plt.show()

# how to use 'lambda' to create an "anonymous" function - just a pithier way of writing functions in Python
g = lambda w: np.tanh(w)

# create the derivative/gradient function of g --> called dgdw
dgdw = value_and_grad(g)

# evaluate the gradient function at a point
w_val = 1.0
print (dgdw(1.0))

# create space over which to evaluate function and gradient
w_vals = np.linspace(-5,5,200)

# evaluate gradient over input range
g_vals = [dgdw(v)[0] for v in w_vals]
dg_vals = [dgdw(v)[1] for v in w_vals]

# create figure
fig = plt.figure(figsize = (7,3))

# plot the function and derivative
plt.plot(w_vals,g_vals,linewidth=2)
plt.plot(w_vals,dg_vals,linewidth=2)
plt.legend(['$g(w)$',r'$\frac{\mathrm{d}}{\mathrm{d}w}g(w)$',r'$\frac{\mathrm{d}^2}{\mathrm{d}w^2}g(w)$'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 13)
plt.show()

# create area over which to evaluate everything
w = np.linspace(-5,5,200); w_0 = 1.0; w_=np.linspace(-2+w_0,2+w_0,200);

# define and evaluate the function, define derivative
g = lambda w: np.tanh(w); dgdw = grad(g);
gvals = [g(v) for v in w]

# create tangent line at a point w_0
tangent = g(w_0) + dgdw(w_0)*(w_ - w_0)

# plot the function and derivative 
fig = plt.figure(figsize = (6,4))
plt.plot(w,gvals,c = 'k',linewidth=2,zorder = 1)
plt.plot(w_,tangent,c = [0,1,0.25],linewidth=2,zorder = 2)
plt.scatter(w_0,g(w_0),c = 'r',s=50,zorder = 3,edgecolor='k',linewidth=1)
plt.legend(['$g(w)$','tangent'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 13)
plt.show()

# create area over which to evaluate everything
w = np.linspace(-5,5,200); w_0 = 1.0; w_=np.linspace(-2+w_0,2+w_0,200);

# define and evaluate the function, define derivative
g = lambda w: np.tanh(w); dgdw = grad(g); dgdw2 = grad(dgdw);
gvals = [g(v) for v in w]

# create tangent line and quadratic
tangent = g(w_0) + dgdw(w_0)*(w_ - w_0)
quadratic = g(w_0) + dgdw(w_0)*(w_ - w_0) + 0.5*dgdw2(w_0)*(w_ - w_0)**2

# plot the function and derivative 
fig = plt.figure(figsize = (7,4))
plt.plot(w,gvals,c = 'k',linewidth=2,zorder = 1)
plt.plot(w_,tangent,c = [0,1,0.25],linewidth=2,zorder = 2)
plt.plot(w_,quadratic,c = [0,0.75,1],linewidth=2,zorder = 2)
plt.scatter(w_0,g(w_0),c = 'r',s=50,zorder = 3,edgecolor='k',linewidth=1)
plt.legend(['$g(w)$','tangent line','tangent quadratic'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 13)
plt.show()

def my_func(w):
    return np.tanh(w[0] - w[1])

# evaluate our multi-input function at a random point
w_val = np.random.randn(2,1)
my_func(w_val)

### evaluate our function over a fine range of points on a square
# produce grid of values
s = np.linspace(-5,5,200)
w1,w2 = np.meshgrid(s,s)

# reshape grid and evaluate all points using our function
w1 = np.reshape(w1,(1,np.size(w1)))
w2 = np.reshape(w2,(1,np.size(w2)))
w = np.concatenate((w1,w2),axis = 0)
g_vals = my_func(w)


# generate figure and panel
fig = plt.figure(figsize = (4,4))
ax = fig.gca(projection='3d')

# re-shape inputs and output for plotting
w1 = np.reshape(w1,(np.size(s),np.size(s)))
w2 = np.reshape(w2,(np.size(s),np.size(s)))
g_vals = np.reshape(g_vals,(np.size(s),np.size(s)))

# Plot the surface
ax.plot_surface(w1,w2,g_vals,alpha = 0.2,color = 'r')
ax.view_init(10,50)
plt.show()

# compute the second derivative of our input function
nabla_g = grad(my_func)
nabla_vals = np.array([nabla_g(v) for v in w.T])

# separate out each partial derivative from the gradient evaluations
partial_1vals = nabla_vals[:,0]
partial_2vals = nabla_vals[:,1]

# reshape each partial evaluations appropriately for plotting
partial_1vals = np.reshape(partial_1vals,(np.size(s),np.size(s)))
partial_2vals = np.reshape(partial_2vals,(np.size(s),np.size(s)))

# load in the gridspec tool from matplotlib for better subplot handling
from matplotlib import gridspec

# initialize figure
fig = plt.figure(figsize = (11,3))

# create subplot with 1 panel
gs = gridspec.GridSpec(1,3) 
ax1 = plt.subplot(gs[0],projection = '3d'); 
ax2 = plt.subplot(gs[1],projection = '3d'); 
ax3 = plt.subplot(gs[2],projection = '3d'); 

# plot surfaces
ax1.plot_surface(w1,w2,g_vals,alpha = 0.25,color = 'r')
ax1.set_title(r'$g\left(\mathbf{w}\right)$',fontsize = 12)
ax1.view_init(10,50)

ax2.plot_surface(w1,w2,partial_1vals,alpha = 0.25,color = 'r')
ax2.set_title(r'$\frac{\partial}{\partial w_1}g\left(\mathbf{w}\right)$',fontsize = 12)
ax2.view_init(10,50)

ax3.plot_surface(w1,w2,partial_2vals,alpha = 0.25,color = 'r') 
ax3.set_title(r'$\frac{\partial}{\partial w_2}g\left(\mathbf{w}\right)$',fontsize = 12)
ax3.view_init(10,50)
plt.show()