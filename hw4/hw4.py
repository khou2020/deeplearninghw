# imports necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from PIL import Image

# this is needed to compensate for matplotlib notebook's tendancy to blow up images when plotted inline
from matplotlib import rcParams
rcParams['figure.autolayout'] = True

def extract_patches(image_list, number_of_patches, patch_size):
    import random

    imgs = []
    for path in image_list:
        imgs.append(np.array(Image.open(path).convert('L')))

    patches = np.empty([patch_size * patch_size, number_of_patches], int)

    for i in range(number_of_patches):
        img = imgs[(int)(i * len(image_list) / number_of_patches)]
        while True:
            u = random.randint(0, img.shape[0] - patch_size)
            v = random.randint(0, img.shape[0] - patch_size)
            patch = img[u:u + patch_size, v: v + patch_size].reshape([patch_size * patch_size])
            if np.std(patch) >= 0.1:
                break
        patches[:, i] = patch

    return patches

image_1 = 'images/bean.jpg'
image_2 = 'images/dog.jpg'
image_3 = 'images/flyer.jpg'
image_4 = 'images/Trey_Matt.png'

image_list = [image_1, image_2, image_3, image_4]
number_of_patches = 100000
patch_size = 12
              
patches = extract_patches(image_list, number_of_patches, patch_size)  

def show_images(X):
    '''
    Function for plotting input images, stacked in columns of input X.
    '''
    # plotting mechanism taken from excellent answer from stack overflow: https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
    plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(10, 10)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 
    
    # shape of square version of image
    square_shape = int((X.shape[0])**(0.5))

    for i in range(min(100,X.shape[1])):
        # plot image in panel
        ax = plt.subplot(gs1[i])
        im = ax.imshow(np.reshape(X[:,i],(square_shape,square_shape)),cmap = 'gray')

        # clean up panel
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()
    
# Plot the first 100 patches 
show_images(patches)

# compute eigendecomposition of data covariance matrix for PCA transformation
def PCA(x,**kwargs):
    # regularization parameter for numerical stability
    lam = 10**(-7)
    if 'lam' in kwargs:
        lam = kwargs['lam']

    # create the correlation matrix
    P = float(x.shape[1])
    Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

    # use numpy function to compute eigenvalues / vectors of correlation matrix
    d,V = np.linalg.eigh(Cov)
    return d,V

# ZCA-sphereing - use ZCA to normalize input features
def ZCA_sphereing(x,**kwargs):
    # Step 1: mean-center the data
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_centered = x - x_means

    # Step 2: compute pca transform on mean-centered data
    d,V = PCA(x_centered,**kwargs)

    # Step 3: divide off standard deviation of each (transformed) input, 
    # which are equal to the returned eigenvalues in 'd'.  
    stds = (d[:,np.newaxis])**(0.5)
    normalizer = lambda data: np.dot(V, np.dot(V.T,data - x_means)/stds)

    # create inverse normalizer
    inverse_normalizer = lambda data: np.dot(V,np.dot(V.T, data)*stds) + x_means

    # return normalizer 
    return normalizer,inverse_normalizer

normalizer, inverse_normalizer = ZCA_sphereing(patches)

patches_ZCA_normalized = normalizer(patches)

show_images(patches_ZCA_normalized)

# perform K-means clustering
from sklearn.cluster import KMeans

# number of clusters
num_clusters = 100 

clusterer = KMeans(n_clusters=num_clusters, max_iter = 2000, n_init = 1)

# fit the algorithm to our dataset
clusterer.fit(patches_ZCA_normalized.T)

# extract cluster centroids
centroids = clusterer.cluster_centers_.T

show_images(centroids)

# YOUR CODE GOES HERE

clusterer = KMeans(n_clusters=num_clusters, max_iter = 2000, n_init = 1)

# fit the algorithm to our dataset
clusterer.fit(patches.T)

# extract cluster centroids
centroids = clusterer.cluster_centers_.T

show_images(centroids)