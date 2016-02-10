import sys
import time

import numpy as np


def predict(model, corpus, sample_x, sample_y, posits):

    print '\nTEST'
    print '\tIndex: ',
    start = time.time()

    clusters = []

    correct = 0.
    correct_t = 0.
    correct_f = 0.
    total = 0.
    total_t = 0.
    k = 0

    for doc_index in xrange(len(sample_x)):
        cluster = []
        d_sample_x = sample_x[doc_index]
        d_sample_y = sample_y[doc_index]
        d_posits = posits[doc_index]

        for m_index in xrange(len(d_sample_x)):
            if k % 1000 == 0 and k != 0:
                print '%d' % k,
                sys.stdout.flush()

            _sample_x = d_sample_x[m_index]
            _sample_y = d_sample_y[m_index]
            posit_i = d_posits[m_index]

            c, y_hat, y_p = model(_sample_x, _sample_y)

            correct += np.sum(c)
            total += len(_sample_y)
            total_t += np.sum(_sample_y)

            cluster = add_to_cluster(cluster, y_hat, y_p, posit_i)

            for u in zip(c, _sample_y):
                if u[0] == 1:
                    if u[1] == 1:
                        correct_t += 1
                    else:
                        correct_f += 1

            k += 1
        clusters.append(cluster)

    end = time.time()

    total_f = total - total_t
    accuracy = correct / total
    accuracy_t = correct_t / total_t
    accuracy_f = correct_f / total_f

    print '\n\tTime: %f seconds' % (end - start)
    print '\tAcc Total:     %f\tCorrect: %d\tTotal: %d' % (accuracy, correct, total)
    print '\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (accuracy_t, correct_t, total_t)
    print '\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (accuracy_f, correct_f, total_f)

    output(fn='result.txt', corpus=corpus, clusters=clusters)


def add_to_cluster(cluster, mention_pair_index, mention_pair_p, posit_i):
    """
    :param cluster: 1D: n_cluster; list, 2D: n_mentions; set
    :param mention_pair_index:
    :param posit_i:
    :return:
    """

    if mention_pair_p < 0.5:
        return cluster

    posit = posit_i[mention_pair_index]  # posit: (sent_i, span, sent_j, span)
    ant = (posit[0], posit[1])
    ment = (posit[2], posit[3])

    for c in cluster:
        if ant in c:
            c.add(ment)
    else:
        cluster.append(set([ant, ment]))

    return cluster


def output(fn, corpus, clusters):
    with open(fn, 'w') as f:
        for doc, cluster in zip(corpus, clusters):
            cluster.sort()
            for c_i, c in enumerate(cluster):
                for m in c:
                    sent_i = m[0]
                    span = m[1]
                    sent = doc[sent_i]

                    for j in xrange(span[0], span[1]+1):
                        w = sent[j]
                        sent[j] = (w[0], w[1], w[2], w[3], w[4], w[5], w[6], str(c_i))

        for doc in corpus:
            for sent in doc:
                for i, w in enumerate(sent):
                    print >> f, '%s\t%s\t%d\t%s\t%s' % (w[0].encode('utf-8'), w[1].encode('utf-8'), i, w[2].encode('utf-8'), w[-1].encode('utf-8'))
                print >> f
