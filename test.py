__author__ = 'hiroki'

import sys
import time


def predict(model, samples, indices, gold_mentions, gold_corefs):
    """
    :param: samples: 1D: n_doc, 2D: n_sents * n_cand_mentions, 2D: word_dim * 2
    :param: indices: 1D: n_doc, 2D: n_sents * n_cand_mentions; elem=((sent_index, (i, j)),(sent_index, (i, j)))
    :param: gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(i, j)
    :param: gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=coref_id
    """

    print '\tIndex: ',
    start = time.time()
    correct = 0
    total = 0.

    for d_index in xrange(len(samples)):
        if d_index % 100 == 0 and d_index != 0:
            print '%d' % d_index,
            sys.stdout.flush()

        for index in xrange(len(samples[d_index])):

            sample = samples[d_index][index]

            if len(sample) == 0:
                continue

            max_pair_index, max_pair_prob = model(sample)
            m1, m2 = indices[d_index][index][max_pair_index]

            g1 = None
            g2 = None
            for g_ment, g_coref in zip(gold_mentions[d_index][m1[0]], gold_corefs[d_index][m1[0]]):
                if m1[1] == g_ment:
                    g1 = g_coref
            for g_ment, g_coref in zip(gold_mentions[d_index][m2[0]], gold_corefs[d_index][m2[0]]):
                if m2[1] == g_ment:
                    g2 = g_coref

            if g1 and g2 and g1 == g2:
                correct += 1

            total += 1.

    end = time.time()

    print '\tTime: %f seconds' % (end - start)
    print '\tAccuracy: %f\tCorrect: %d\tTotal: %d' % ((correct / total), correct, total)




