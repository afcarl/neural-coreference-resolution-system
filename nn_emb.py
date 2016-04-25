import numpy as np
import theano
import theano.tensor as T

from nn_utils import sample_weights, L2_sqr, sigmoid, relu
from optimizers import sgd


class Model(object):
    def __init__(self, x_span, x_word, x_ctx, x_dist, x_slen, y, init_emb, n_vocab, dim_w, dim_d, dim_h, L2_reg):
        """
        :param x_span: 1D: batch, 2D: limit * 2 (10); elem=word id
        :param x_word: 1D: batch, 2D: 4 (m_first, m_last, a_first, a_last); elem=word id
        :param x_ctx : 1D: batch, 2D: window * 2 * 2 (20); elem=word id
        :param x_dist: 1D: batch; 2D: 2; elem=[sent dist, ment dist]
        :param x_slen: 1D: batch; 2D: 3; elem=[m_span_len, a_span_len, head_match]
        :param y     : 1D: batch
        """

        self.input  = [x_span, x_word, x_ctx, x_dist, y]
        self.x_span = x_span
        self.x_word = x_word
        self.x_ctx  = x_ctx
        self.x_dist = x_dist
        self.x_slen = x_slen
        self.y      = y

        dim_x = dim_w * (10 + 4 + 4 + 2 + 3)
        batch = y.shape[0]

        """ Params """
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab, dim_w))
        else:
            self.emb = theano.shared(init_emb)

        self.W_d = theano.shared(sample_weights(dim_d, dim_w))
        self.W_l = theano.shared(sample_weights(7, dim_w))
        self.W_i = theano.shared(sample_weights(dim_x, dim_h))
        self.W_h = theano.shared(sample_weights(dim_h, dim_h))
        self.W_o = theano.shared(sample_weights(dim_h))
        self.params = [self.W_d, self.W_l, self.W_i, self.W_h, self.W_o]

        """ Input Layer """
        x_vec = T.concatenate([x_span, x_word, x_ctx], 1).flatten()  # 1D: batch * (limit * 2 + 4 + 20)
        x_in = self.emb[x_vec]     # 1D: batch, 2D: limit * 2, 3D: dim_w
        x_d = self.W_d[x_dist]     # 1D: batch, 2D: 2, 3D: dim_w
        x_l = self.W_l[x_slen]     # 1D: batch, 2D: 2, 3D: dim_w
        x = T.concatenate([x_in.reshape((batch, -1)), x_d.reshape((batch, -1)), x_l.reshape((batch, -1))], 1)

        """ Intermediate Layers """
        h1 = relu(T.dot(x, self.W_i))   # h1: 1D: batch, 2D: dim_h
        h2 = relu(T.dot(h1, self.W_h))  # h2: 1D: batch, 2D: dim_h

        """ Output Layer """
        p_y = sigmoid(T.dot(h2, self.W_o))  # p_y: 1D: batch

        """ Cost Function """
        self.nll = - T.sum(y * T.log(p_y) + (1. - y) * T.log((1. - p_y)))  # TODO: ranking criterion
        self.cost = self.nll + L2_reg * L2_sqr(params=self.params) / 2

        """ Update """
        self.updates = sgd(self.cost, self.params, self.emb, x_in)

        """ Predicts """
        self.thresholds = theano.shared(np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=theano.config.floatX))
        self.y_hat = self.binary_predict(p_y)  # 1D: batch, 2D: 9 (thresholds)
        self.y_hat_index = T.argmax(p_y)
        self.p_y_hat = p_y[self.y_hat_index]

        """ Check Results """
        self.result = T.eq(self.y_hat, y.reshape((y.shape[0], 1)))  # 1D: batch, 2D: 9 (thresholds)
        self.total_p = T.sum(self.y_hat, 0)
        self.total_r = T.sum(y, keepdims=True)
        self.correct = T.sum(self.result, 0)
        self.correct_t, self.correct_f = correct_tf(self.result, y.reshape((y.shape[0], 1)))

    def binary_predict(self, p_y):
        p_y = T.repeat(p_y.reshape((p_y.shape[0], 1)), 9, 1)
        return T.switch(p_y >= self.thresholds, 1, 0)  # TODO: threshold


def correct_tf(result, y):
    correct_t = T.sum(T.switch(result, T.eq(y, 1), 0), 0)
    correct_f = T.sum(T.switch(result, T.eq(y, 0), 0), 0)
    return correct_t, correct_f

