import numpy as np
import matplotlib.pyplot as plt
    
def plotProgress(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
def plotCompDecode(x, decode, n = 10, xNoise = None):
    '''
    n: how many digits we will display
    '''
    if xNoise is not None:    
        plt.figure(figsize=(20, 2))
        for i in range(n):
            # display original with noise
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(xNoise[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decode[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def plotScatterEncode(encode, y, xlim, ylim, numShow = 2000, sizeFont=40):
    plt.figure(figsize=(12, 12))
    plt.scatter(encode[0:numShow, 0], encode[0:numShow, 1], c=y[0:numShow], cmap='viridis')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.colorbar()
    for i in np.unique(y, return_index=True)[1]: # list the first indices of each digit 
        plt.text(encode[i, 0], encode[i, 1], y[i], fontsize=sizeFont)
    plt.show()
    

def plotScatterDecode(decoder, sizeDigit, xlim, ylim, numDigit = 15):
    figure = np.zeros((sizeDigit * numDigit, sizeDigit * numDigit))
    # we will sample n points within 4 times of the standard deviations
    xGrid = np.linspace(xlim[0], xlim[1], numDigit)
    yGrid = np.linspace(ylim[0], ylim[1], numDigit)

    for i, xi in enumerate(xGrid):
        for j, yi in enumerate(yGrid):
            zSample = np.array([[xi, yi]])
            decode = decoder.predict(zSample)
            digit = decode[0].reshape(sizeDigit, sizeDigit)
            figure[j * sizeDigit: (j + 1) * sizeDigit,  # 1st pos for y
                   i * sizeDigit: (i + 1) * sizeDigit] = np.flipud(digit)

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='viridis', origin='lower')
    plt.show()
    
def addNoise(x, factNoise = 0.5, std = 1., mean = 0):
    xNoise = x + factNoise * np.random.normal(loc=mean, scale=std, size=x.shape) 
    xNoise = np.clip(xNoise, 0., 1.)
    return xNoise