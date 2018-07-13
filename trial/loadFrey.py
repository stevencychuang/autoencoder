import random
numSeed = 42
random.seed(42)
import os
import numpy as np
np.random.seed(numSeed)
from urllib.request import urlretrieve

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print('Downloading {}'.format(filename))
    urlretrieve(source + filename, filename)
    
def load_frey_face_images(filename):
    if not os.path.exists(filename):
        download(filename, source='http://www.cs.nyu.edu/~roweis/data/')
    import scipy.io as sio
    data = sio.loadmat(filename)['ff'].reshape(28, 20, -1).transpose(2, 0, 1)
    return data / np.float32(255)


def load_frey_face_dataset():
    X = load_frey_face_images('frey_rawface.mat')
    np.random.shuffle(X)
    X_train, X_val = X[:-565], X[-565:]
    return X_train, X_val