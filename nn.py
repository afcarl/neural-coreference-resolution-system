import theano
import theano.tensor as T

from nn_utils import sample_weights, L2_sqr, sigmoid, tanh, relu
from optimizers import adam


class Model(object):
    def __init__(self, x_word, x_sdist, y, init_emb, n_vocab, dim_x_word, dim_x_sdist, dim_h, n_label, L2_reg):
        """
        :param x_word: 1D: n_ants, 2D: window * 2; elem=word id
        :param x_sdist: 1D: n_ants; elem=distance between sentences of ant and ment
        :param y: 1D: n_ants
        :return:
        """

        self.x_word = x_word
        self.x_sdist = x_sdist
        self.y = y
        self.input = [self.x_word, self.x_sdist, self.y]

        self.init_emb = init_emb
        self.n_vocab = n_vocab
        self.dim_x_word = dim_x_word
        self.dim_x_sdist = dim_x_sdist
        self.dim_h = dim_h
        self.n_label = n_label
        self.L2_reg = L2_reg
        self.dim_phi = dim_x_word * 10

        """ Params """
        if self.init_emb is None:
            self.emb = theano.shared(sample_weights(self.n_vocab, self.dim_x_word))
        else:
            self.emb = theano.shared(self.init_emb)

        self.W_sdist = theano.shared(sample_weights(self.dim_x_sdist, self.dim_h))
        self.W_in = theano.shared(sample_weights(self.dim_phi, self.dim_h))
        self.W_h = theano.shared(sample_weights(self.dim_h, self.dim_h))
        self.W_out = theano.shared(sample_weights(self.dim_h, self.n_label))
        self.params = [self.W_sdist, self.W_in, self.W_h, self.W_out]

        """ Network """
        x_word_in = self.emb[x_word]  # x_word_in: 1D: n_ants, 2D: window * 2, 3D: dim_x
        x_sdist_in = self.W_sdist[x_sdist]  # x_sdist_in: 1D: n_ants, 2D: dim_hidden
        x_word_in_reshape = x_word_in.reshape((x_word_in.shape[0], -1))

#        h1 = tanh(T.dot(T.concatenate([x_word_in_reshape, x_sdist_in_reshape], 1), self.W_in))  # h1: 1D: batch, 2D: dim_h
        h1 = tanh(T.dot(x_word_in_reshape, self.W_in) + x_sdist_in)  # h1: 1D: n_ants, 2D: dim_h
        h2 = tanh(T.dot(h1, self.W_h))  # h2: 1D: n_ants, 2D: dim_h
        self.p_y = sigmoid(T.dot(h2, self.W_out)).flatten()  # p_y: 1D: n_ants

        """ Predicts """
        self.y_hat_index = T.argmax(self.p_y, axis=0)
        self.y_hat_p = self.p_y[self.y_hat_index]
        self.y_pair_pred = binary_predict(self.p_y)

        """ Cost Function """
        self.nll = - T.sum(self.y * T.log(self.p_y) + (1. - self.y) * T.log((1. - self.p_y)))
        self.cost = self.nll + L2_reg * L2_sqr(params=self.params) / 2

        """ Update """
        self.g = T.grad(self.cost, self.params)
        self.updates = adam(self.params, self.g)

        """ Check the Accuracy """
        self.correct = T.eq(self.y_pair_pred, self.y)


def binary_predict(p_y):
    return T.switch(p_y >= 0.5, 1, 0)

