"""
Created on 2018/05/01
Revised on 2018/10/26

@author: STEVEN.CY.CHUANG
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import metrics
import matplotlib.pyplot as plt
    
def plot_progress(history):
    """
    Plot the training progress.
    args:
        history (keras History object)
    """
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    
def comp_reconst(x, decode, method="rmse"):
    # Flatten features as one vector
    num_inst = x.shape[0]
    x_resh = x.reshape(num_inst,-1)
    decode_resh = decode.reshape(num_inst,-1)
    
    # Determine the error by methods
    if method=="rmse":
        err = np.sqrt(mean_squared_error(x_resh, decode_resh))
    elif method=="mse":
        err = mean_squared_error(x_resh, decode_resh)
    elif method=="mae":
        err = mean_absolute_error(x_resh, decode_resh)
    elif method=="log_loss":
        err = -np.sum(x_resh*np.log(decode_resh))/num_inst    
    return err
        
    
def plot_comp_decode(x, decode, n=10, x_noise=None, size_digit=None):
    """
    This function is to plot the comparison among 
    1. the original image
    2. the reconstructed image
    3. the original image with disturbance (if there is this argument)
    args:
        x (numpy ndarray): the vector of the original image.
        decode (numpy ndarray): the vector of the reconstructed image.
        n (int): how many digits we will display. Default is 10.
        x_noise (numpy ndarray): the vector of the original image with disturbance.
        size_digit (tuple): the exact size of image pixels. the default value None imply the image is squared with grayscale(shape is (m, m)).
    """
    # Check if the size to reshape can be squared
    if size_digit is None:
        width = int(np.sqrt(x.shape[1]))
        if width * width != x.shape[1]:
            raise ValueError("the size of pixel is not squared!")
        size_digit = (width, width)
    
    # Check if there is the input for x with nosing value
    if x_noise is not None:    
        plt.figure(figsize=(20, 2))
        for i in range(n):
            # display original with noise
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(x_noise[i].reshape(size_digit))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    # Plot for the original and decoded pictures  
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x[i].reshape(size_digit))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decode[i].reshape(size_digit))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    
def plot_scatter_encode(encode, y, xlim, ylim, num_show=2000, size_font=40, dim_show=[0, 1], size_marker=12):
    """
    This function is to plot the scatter for the encoded x of the dataset.
    It should be noted that the plot always show the first 2 dimensions of x.
    args:
        encode (numpy ndarray): the encoded x of instances.
        y (list; numpy ndarray): the label of instances.
        xlim (tuple): the limit to show the 1st dimension of x.
        ylim (tuple): the limit to show the 2nd dimension of x.
        num_show (int): the number of instances to show. Default is 2000.
        size_font (int): the font size for the labels of instances if they exist. Default is 40.
        dim_show (list): the two dimensions to show encode. Default is [0, 1].
        size_marker (int): the marker size for each instance of scatter. Default is 12.
    """
    s = [size_marker for n in range(len(encode))]
    plt.figure(figsize=(12, 12))
    if y is None:
        plt.scatter(encode[0:num_show, dim_show[0]], encode[0:num_show, dim_show[1]], cmap="viridis", s=s)
    else:
        plt.scatter(encode[0:num_show, dim_show[0]], encode[0:num_show, dim_show[1]], c=y[0:num_show], cmap="viridis", s=s)
        plt.colorbar()
        for i in np.unique(y, return_index=True)[1]:  # list the first indices of each digit 
            plt.text(encode[i, dim_show[0]], encode[i, dim_show[1]], y[i], fontsize=size_font)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.show()
    

def plot_scatter_decode(decoder, size_digit, xlim, ylim, num_digit=15, dim_show=[0, 1]):
    """
    This function is to plot the reconstructed images from the sampling grid of encoded dimension.
    It should be noted that the sampling grid is only for the first 2 encoded dimensions.
    args:
        decoder (keras model): the generator for reconstructed image 
        size_digit: the exact size of image pixels
        xlim: the limit to show the 1st encoded dimension
        ylim: the limit to show the 2nd encoded dimension
        num_digit: the number of reconstructed image in one dimension
        dim_show (list): the two dimensions to show encode. Default is [0, 1].
    """
    # Check the picutres with grayscale (m1, m2) or RGB (m1, m2, c)
    if len(size_digit) > 2:
        figure = np.zeros((size_digit[0] * num_digit, size_digit[1] * num_digit, size_digit[2]))
    else:
        figure = np.zeros((size_digit[0] * num_digit, size_digit[1] * num_digit))
        
    # we will sample n points within the limits of the first 2 dimensions
    x_grid = np.linspace(xlim[0], xlim[1], num_digit)
    y_grid = np.linspace(ylim[0], ylim[1], num_digit)

    # Get the number of encoded dimension
    dim_encode = decoder.layers[0].input_shape[1]
    
    # Plot each generated digit from the grid sampling
    for i, xi in enumerate(x_grid):
        for j, yi in enumerate(y_grid):
            z_sample = np.array([np.zeros(dim_encode)])  # generate the sample whose shape is (1, dim_encode) and all values are 0
            z_sample[:, dim_show] = np.array([xi, yi])  # just replace the first 2 dimension with the sampling grid
            decode = decoder.predict(z_sample)
            digit = decode[0].reshape(size_digit)
            figure[j * size_digit[0]: (j + 1) * size_digit[0],  # 1st pos for y
                   i * size_digit[1]: (i + 1) * size_digit[1]] = np.flipud(digit)

    plt.figure(figsize=(20, 20))
    plt.imshow(figure, cmap="viridis", origin="lower")
    plt.show()