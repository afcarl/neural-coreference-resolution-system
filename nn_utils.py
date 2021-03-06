import numpy as np
import theano
import theano.tensor as T


def relu(x):
    return T.nnet.relu(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def get_zeros(shape):
    return np.zeros(shape, dtype=theano.config.floatX)


def build_shared_zeros(shape):
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        borrow=True
    )


def sample_weights(size_x, size_y=0):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / size_x),
                                         high=np.sqrt(6.0 / size_x),
                                         size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    return W


def L2_sqr(params):
    sqr = 0.0
    for p in params:
        sqr += (p ** 2).sum()
    return sqr
