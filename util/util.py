import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import metrics
import matplotlib.pyplot as plt
    
def plotProgress(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

def compReconst(x, decode, method='rmse'):
    # Flatten features as one vector
    numInst = x.shape[0]
    xResh = x.reshape(numInst,-1)
    decodeResh = decode.reshape(numInst,-1)
    
    # Determine the error by methods
    if method=='rmse':
        err = np.sqrt(mean_squared_error(xResh, decodeResh))
    elif method=='mse':
        err = mean_squared_error(xResh, decodeResh)
    elif method=='mae':
        err = mean_absolute_error(xResh, decodeResh)
    elif method=='log_loss':
        err = -np.sum(xResh*np.log(decodeResh))/numInst    
    return err
        
    
def plotCompDecode(x, decode, n=10, xNoise=None, sizeDigit=None):
    '''
    This function is to plot the comparison among 
    1. the original image
    2. the reconstructed image
    3. the original image with disturbance (if there is this argument)
    args:
        x: the vector of the original image
        decode: the vector of the reconstructed image
        n: how many digits we will display
        xNoise: the vector of the original image with disturbance
        sizeDigit: the exact size of image pixels. the default value None imply the image is squared with grayscale(shape is (m, m))
    '''
    # Check if the size to reshape can be squared
    if sizeDigit is None:
        width = int(np.sqrt(x.shape[1]))
        if width * width != x.shape[1]:
            raise ValueError('the size of pixel is not squared!')
        sizeDigit = (width, width)
    
    # Check if there is the input for x with nosing value
    if xNoise is not None:    
        plt.figure(figsize=(20, 2))
        for i in range(n):
            # display original with noise
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(xNoise[i].reshape(sizeDigit))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    # Plot for the original and decoded pictures  
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x[i].reshape(sizeDigit))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decode[i].reshape(sizeDigit))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def plotScatterEncode(encode, y, xlim, ylim, numShow=2000, sizeFont=40, dimShow=[0, 1], markersize=12):
    '''
    This function is to plot the scatter for the encoded x of the dataset.
    It should be noted that the plot always show the first 2 dimensions of x.
    args:
        encode: the encoded x of instances
        y: the label of instances
        xlim: the limit to show the 1st dimension of x
        ylim: the limit to show the 2nd dimension of x
        numShow: the number of instances to show
        sizeFont: the font size for the labels of instances if they exist
    '''
    s = [markersize for n in range(len(encode))]
    plt.figure(figsize=(12, 12))
    if y is None:
        plt.scatter(encode[0:numShow, dimShow[0]], encode[0:numShow, dimShow[1]], cmap='viridis', s=s)
    else:
        plt.scatter(encode[0:numShow, dimShow[0]], encode[0:numShow, dimShow[1]], c=y[0:numShow], cmap='viridis', s=s)
        plt.colorbar()
        for i in np.unique(y, return_index=True)[1]:  # list the first indices of each digit 
            plt.text(encode[i, dimShow[0]], encode[i, dimShow[1]], y[i], fontsize=sizeFont)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.show()
    

def plotScatterDecode(decoder, sizeDigit, xlim, ylim, numDigit=15, dimShow=[0, 1]):
    '''
    This function is to plot the reconstructed images from the sampling grid of encoded dimension.
    It should be noted that the sampling grid is only for the first 2 encoded dimensions.
    args:
        decoder: the generator for reconstructed image 
        sizeDigit: the exact size of image pixels
        xlim: the limit to show the 1st encoded dimension
        ylim: the limit to show the 2nd encoded dimension
        numDigit: the number of reconstructed image in one dimension
    '''
    # Check the picutres with grayscale (m1, m2) or RGB (m1, m2, c)
    if len(sizeDigit) > 2:
        figure = np.zeros((sizeDigit[0] * numDigit, sizeDigit[1] * numDigit, sizeDigit[2]))
    else:
        figure = np.zeros((sizeDigit[0] * numDigit, sizeDigit[1] * numDigit))
        
    # we will sample n points within the limits of the first 2 dimensions
    xGrid = np.linspace(xlim[0], xlim[1], numDigit)
    yGrid = np.linspace(ylim[0], ylim[1], numDigit)

    # Get the number of encoded dimension
    dimEncode = decoder.layers[0].input_shape[1]
    
    # Plot each generated digit from the grid sampling
    for i, xi in enumerate(xGrid):
        for j, yi in enumerate(yGrid):
            zSample = np.array([np.zeros(dimEncode)])  # generate the sample whose shape is (1, dimEncode) and all values are 0
            zSample[:, dimShow] = np.array([xi, yi])  # just replace the first 2 dimension with the sampling grid
            decode = decoder.predict(zSample)
            digit = decode[0].reshape(sizeDigit)
            figure[j * sizeDigit[0]: (j + 1) * sizeDigit[0],  # 1st pos for y
                   i * sizeDigit[1]: (i + 1) * sizeDigit[1]] = np.flipud(digit)

    plt.figure(figsize=(20, 20))
    plt.imshow(figure, cmap='viridis', origin='lower')
    plt.show()
    
def addNoise(x, factNoise=0.5, std=1., mean=0):
    xNoise = x + factNoise * np.random.normal(loc=mean, scale=std, size=x.shape) 
    xNoise = np.clip(xNoise, 0., 1.)
    return xNoise
