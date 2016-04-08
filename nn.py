import theano
import theano.tensor as T

from nn_utils import sample_weights, L2_sqr, sigmoid, tanh, relu
from optimizers import adam


class Model(object):
    def __init__(self, x_span, x_word, x_ctx, x_dist, y, init_emb, n_vocab, dim_w, dim_d, dim_h, L2_reg):
        """
        :param x_span: 1D: batch, 2D: limit * 2 (10); elem=word id
        :param x_word: 1D: batch, 2D: 4 (m_first, m_last, a_first, a_last); elem=word id
        :param x_ctx : 1D: batch, 2D: window * 2 * 2 (20); elem=word id
        :param x_dist: 1D: batch; elem=distance between sentences of ant and ment
        :param y     : 1D: batch
        """

        self.input  = [x_span, x_word, x_ctx, x_dist, y]
        self.x_span = x_span
        self.x_word = x_word
        self.x_ctx  = x_ctx
        self.x_dist = x_dist
        self.y      = y

        dim_x = dim_w * (10 + 4 + 20) + 1
        batch = y.shape[0]

        """ Params """
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab, dim_w))
        else:
            self.emb = theano.shared(init_emb)

        self.W_d = theano.shared(sample_weights(dim_d))
        self.W_i = theano.shared(sample_weights(dim_x, dim_h))
        self.W_h = theano.shared(sample_weights(dim_h, dim_h))
        self.W_o = theano.shared(sample_weights(dim_h))
        self.params = [self.W_d, self.W_i, self.W_h, self.W_o]

        """ Input Layer """
        x_s = self.emb[x_span]     # 1D: batch, 2D: limit * 2,      3D: dim_w
        x_w = self.emb[x_word]     # 1D: batch, 2D: 4,              3D: dim_w
        x_c = self.emb[x_ctx]      # 1D: batch, 2D: window * 2 * 2, 3D: dim_w
        x_d = self.W_d[x_dist]     # 1D: batch
        x = T.concatenate([x_s.reshape((batch, -1)), x_w.reshape((batch, -1)), x_c.reshape((batch, -1)), x_d.reshape((batch, 1))], 1)

        """ Intermediate Layers """
        h1 = tanh(T.dot(x, self.W_i))   # h1: 1D: batch, 2D: dim_h
        h2 = tanh(T.dot(h1, self.W_h))  # h2: 1D: batch, 2D: dim_h

        """ Output Layer """
        p_y = sigmoid(T.dot(h2, self.W_o))  # p_y: 1D: batch

        """ Predicts """
        self.y_hat = binary_predict(p_y)
        self.y_hat_index = T.argmax(p_y)
        self.p_y_hat = p_y[self.y_hat_index]

        """ Cost Function """
        self.nll = - T.sum(y * T.log(p_y) + (1. - y) * T.log((1. - p_y)))
        self.cost = self.nll + L2_reg * L2_sqr(params=self.params) / 2

        """ Update """
        self.grad = T.grad(self.cost, self.params)
        self.updates = adam(self.params, self.grad)

        """ Check Results """
        self.result = T.eq(self.y_hat, y)
        self.total_p = T.sum(self.y_hat)
        self.total_r = T.sum(y)
        self.correct = T.sum(self.result)
        self.correct_t, self.correct_f = correct_tf(self.result, y)


def binary_predict(p_y):
    return T.switch(p_y >= 0.5, 1, 0)


def correct_tf(result, y):
    correct_t = T.sum(T.switch(result, T.eq(y, 1), 0))
    correct_f = T.sum(T.switch(result, T.eq(y, 0), 0))
    return correct_t, correct_f

