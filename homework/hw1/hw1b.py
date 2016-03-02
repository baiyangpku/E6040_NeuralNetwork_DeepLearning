import os
import PIL
import theano
import copy
from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import theano.tensor as T

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    f, axarr = plt.subplots(4, 4)
    
    for i in range(4):
        for j in range(4):
            D_to_plot = np.reshape(D[:, (i*4+j)], (sz, sz))
            plt.subplot(axarr[i, j])
            plt.imshow(D_to_plot, cmap='gray')
            plt.axis('off')
    f.savefig(imname)
    plt.close(f)


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    X = np.dot(D, c).T
    X = X.reshape(256,256) +X_mn
    plt.subplot(ax)
    plt.imshow(X, cmap='gray')
    plt.axis('off')


if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''
    dirc = 'jaffe/'
    
    files = os.listdir(dirc)
    files.sort()
    no_image = len(files)
    Ims = np.empty((no_image, 256*256),dtype = np.float32)
    i=0
    for name in files :
        
        img = PIL.Image.open('jaffe/'+name)
        img_arr = np.array(img, dtype = np.float32)
        img_arr = img_arr.reshape((1,256*256))
        Ims[i,:] = img_arr / 255.0
        i = i+1


    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    '''

    # Gradient descent algorithm
    MAXstep = 55
    D=np.random.rand(256*256,16)
    theta = 0.06
    lamda = np.zeros(16)
   
    for i in range(0,16):  
        for t in range(0,MAXstep):
            temp=0;
            for j in range(0,i):
                temp += lamda[j] * (np.dot(D[:,j],np.dot(D[:,j].T,D[:,i])))
            new_y= D[:,i] + 2.0* theta* np.dot(X.T,np.dot(X,D[:,i])) -2* theta *temp
            D[:,i] = new_y / np.linalg.norm(new_y)   
        lamda[i] = np.dot(np.dot(np.dot(D[:,i].T, X.T),  X), D[:,i])
 

    c = np.dot(D.T, X.T)

    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')

