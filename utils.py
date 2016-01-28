__author__ = 'hiroki'


import numpy as np
from nn_utils import sample_weights, get_zeros


def get_init_emb(vocab_word, emb_dim):
    emb = sample_weights(size_x=vocab_word.size(), size_y=emb_dim)
    emb[0] = get_zeros((1, emb_dim))
    return np.asarray(emb)


def shuffle(sample_x, sample_y):
    new_x = []
    new_y = []

    indices = [i for i in xrange(len(sample_x))]
    np.random.shuffle(indices)

    for i in indices:
        new_x.append(sample_x[i])
        new_y.append(sample_y[i])

    return new_x, new_y


