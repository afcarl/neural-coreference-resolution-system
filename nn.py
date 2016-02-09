import theano
import theano.tensor as T

from nn_utils import sample_weights, L2_sqr, sigmoid, tanh, relu
from optimizers import adam


class Model(object):
    def __init__(self, x, y, init_emb, n_vocab, dim_x, dim_h, n_label, L2_reg):
        """
        :param x: 1D: batch, 2D: dim_x
        :param y: 1D: batch
        :return:
        """

        self.x = x
        self.y = y
        self.init_emb = init_emb
        self.n_vocab = n_vocab
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.n_label = n_label
        self.L2_reg = L2_reg
        self.dim_phi = dim_x * 10

        """ Params """
        if self.init_emb is None:
            emb = theano.shared(sample_weights(self.n_vocab, self.dim_x))
        else:
            emb = theano.shared(self.init_emb)
        self.W_in = theano.shared(sample_weights(self.dim_phi, self.dim_h))
        self.W_h = theano.shared(sample_weights(self.dim_h, self.dim_h))
        self.W_out = theano.shared(sample_weights(self.dim_h, self.n_label))
        self.params = [self.W_in, self.W_h, self.W_out]

        """ Network """
        x_in = emb[x]  # x_in: 1D: batch, 2D: n_phi, 3D: dim_x
        h1 = tanh(T.dot(x_in.reshape((x_in.shape[0], -1)), self.W_in))  # h1: 1D: batch, 2D: dim_h
        h2 = tanh(T.dot(h1, self.W_h))  # h2: 1D: batch, 2D: dim_h
        self.p_y = sigmoid(T.dot(h2, self.W_out)).flatten()  # p_y: 1D: batch

        """ Predicts """
        self.y_pred = T.argmax(self.p_y, axis=0)
        self.y_pair_pred = binary_predict(self.p_y)

        """ Cost Function """
        self.nll = - T.mean(self.y * T.log(self.p_y) + (1. - self.y) * T.log((1. - self.p_y)))
        self.cost = self.nll + L2_reg * L2_sqr(params=self.params) / 2

        """ Update """
        self.g = T.grad(self.cost, self.params)
        self.updates = adam(self.params, self.g)

        """ Check the Accuracy """
        self.correct = T.eq(self.y_pair_pred, self.y)


def binary_predict(p_y):
    return T.switch(p_y >= 0.5, 1, 0)

