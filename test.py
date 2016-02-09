import sys
import time

import numpy as np


def predict(model, sample_x, sample_y):
    """
    :param: samples: 1D: n_doc, 2D: n_sents * n_cand_mentions, 2D: word_dim * 2
    :param: indices: 1D: n_doc, 2D: n_sents * n_cand_mentions; elem=((sent_index, (i, j)),(sent_index, (i, j)))
    :param: gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(i, j)
    :param: gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=coref_id
    """

    print '\nTEST'
    print '\tIndex: ',
    start = time.time()

    correct = 0.
    correct_t = 0.
    correct_f = 0.
    total = 0.
    total_t = 0.
    k = 0

    for doc_index in xrange(len(sample_x)):
        d_sample_x = sample_x[doc_index]
        d_sample_y = sample_y[doc_index]

        for m_index in xrange(len(d_sample_x)):
            if k % 1000 == 0 and k != 0:
                print '%d' % k,
                sys.stdout.flush()

            _sample_x = d_sample_x[m_index]
            _sample_y = d_sample_y[m_index]

            c = model(_sample_x, _sample_y)

            correct += np.sum(c)
            total += len(_sample_y)
            total_t += np.sum(_sample_y)

            for u in zip(c, _sample_y):
                if u[0] == 1:
                    if u[1] == 1:
                        correct_t += 1
                    else:
                        correct_f += 1

            k += 1

    end = time.time()

    total_f = total - total_t
    accuracy = correct / total
    accuracy_t = correct_t / total_t
    accuracy_f = correct_f / total_f

    print '\n\tTime: %f seconds' % (end - start)
    print '\tAcc Total:     %f\tCorrect: %d\tTotal: %d' % (accuracy, correct, total)
    print '\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (accuracy_t, correct_t, total_t)
    print '\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (accuracy_f, correct_f, total_f)
