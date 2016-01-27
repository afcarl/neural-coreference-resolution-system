__author__ = 'hiroki'


import theano
import theano.tensor as T

from nn_utils import sample_weights, L2_sqr, sigmoid, tanh, relu
from optimizers import adam


class Model(object):
    def __init__(self, x, y, dim_x, dim_h, n_label, L2_reg):
        """
        :param x: 1D: batch, 2D: dim_x
        :param y: 1D: batch
        :return:
        """

        self.x = x
        self.y = y

        """ Params """
        W_in = theano.shared(sample_weights(dim_x, dim_h))
        W_h = theano.shared(sample_weights(dim_h, dim_h))
        W_out = theano.shared(sample_weights(dim_h, n_label))
        params = [W_in, W_h, W_out]

        """ Network """
        h1 = relu(T.dot(x, W_in))  # h1: 1D: batch, 2D: dim_h
        h2 = relu(T.dot(h1, W_h))  # h2: 1D: batch, 2D: dim_h
        p_y = sigmoid(T.dot(h2, W_out)).flatten()  # p_y: 1D: batch

        """ Predicts """
#        self.y_pred = T.argmax(self.log_p_y, axis=1)

        """ Cost Function """
        self.nll = - T.mean(y * T.log(p_y) + (1. - y) * T.log((1. - p_y)))
        cost = self.nll + L2_reg * L2_sqr(params=params) / 2

        """ Update """
        g = T.grad(cost, params)
        self.updates = adam(params, g)

        """ Check the Accuracy """
#        self.correct = T.sum(T.eq(self.y_pred, self.y_r))
