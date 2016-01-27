__author__ = 'hiroki'


import numpy as np


def shuffle(sample_x, sample_y):
    new_x = []
    new_y = []

    indices = [i for i in xrange(len(sample_x))]
    np.random.shuffle(indices)

    for i in indices:
        new_x.append(sample_x[i])
        new_y.append(sample_y[i])

    return new_x, new_y


