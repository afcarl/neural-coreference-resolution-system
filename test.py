import sys
import time

import numpy as np


def predict(epoch, model, corpus, doc_names, sample_x_word, sample_x_sdist, sample_y, posits):
    """
    :param epoch: int
    :param model: model
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words, 4D: (doc_id, part_id, form, tag, syn, ne, coref_id)
    :param doc_names: 1D: n_doc; str
    :param sample_x_word: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants, 4D: window * 2; elem=word id
    :param sample_x_sdist: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants; elem=distance
    :param sample_y: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants; elem=label (0/1)
    :param posits: 1D: n_doc, 2D: n_mentions, 3D: n_mentions_pairs, 4D: (m_sent_i, m_span, a_sent_i, a_span)
    :return:
    """

    print '\nTEST'
    print '\tIndex: ',
    start = time.time()

    clusters = []
    results = []

    correct = 0.
    correct_t = 0.
    correct_f = 0.
    total = 0.
    total_r = 0.
    total_p = 0.
    k = 0

    for doc_index in xrange(len(sample_x_word)):
        cluster = []
        result = []

        d_sample_x_word = sample_x_word[doc_index]
        d_sample_x_sdist = sample_x_sdist[doc_index]
        d_sample_y = sample_y[doc_index]
        d_posits = posits[doc_index]

        for m_index in xrange(len(d_sample_x_word)):
            if k % 1000 == 0 and k != 0:
                print '%d' % k,
                sys.stdout.flush()

            _sample_x_word = d_sample_x_word[m_index]
            _sample_x_sdist = d_sample_x_sdist[m_index]
            _sample_y = d_sample_y[m_index]
            posit_i = d_posits[m_index]

            predict_i, correct_i, y_hat, y_p = model(_sample_x_word, _sample_x_sdist, _sample_y)

            correct += np.sum(correct_i)
            total += len(_sample_y)
            total_r += np.sum(_sample_y)
            total_p += np.sum(predict_i)

            cluster = add_to_cluster(cluster, y_hat, y_p, posit_i)
            result.append((y_p, posit_i[y_hat]))

            for u in zip(correct_i, _sample_y):
                if u[0] == 1:
                    if u[1] == 1:
                        correct_t += 1
                    else:
                        correct_f += 1

            k += 1
        clusters.append(cluster)
        results.append(result)

    end = time.time()

    total_f = total - total_r
    accuracy = correct / total
    accuracy_t = correct_t / total_r
    accuracy_f = correct_f / total_f

    precision = correct_t / total_p
    recall = correct_t / total_r
    f = 2 * precision * recall / (precision + recall)

    print '\n\tTime: %f seconds' % (end - start)
    print '\tAcc Total:     %f\tCorrect: %d\tTotal: %d' % (accuracy, correct, total)
    print '\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (accuracy_t, correct_t, total_r)
    print '\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (accuracy_f, correct_f, total_f)
    print '\tPrecision:     %f\tRecall:  %f\tF1:    %f' % (precision, recall, f)
    print '\tTotal_P:       %d\tTotal_R: %d' % (total_p, total_r)

    output_detail_results(fn='result-output.epoch-%d.txt' % (epoch + 1), corpus=corpus, results=results)
    output_results(fn='result.epoch-%d.txt' % (epoch + 1), corpus=corpus, doc_names=doc_names, clusters=clusters)


def output_detail_results(fn, corpus, results):
    with open(fn, 'w') as f:
        for doc, result in zip(corpus, results):
            mentions = []
            for s_i, sent in enumerate(doc):
                mention = []
                for r in result:  # r: (y_hat, y_p, y_posit=(m_sent_i, m_span, a_sent_i, a_span))
                    posit = r[-1]
                    if posit[0] == s_i:
                        mention.append(r)
                mentions.append(mention)

            for s_i, sent in enumerate(doc):
                result = mentions[s_i]
                print >> f, 'RESULT'
                for r in result:
                    print >> f, '\t%s\tMent:%s-%s\tAnt:%s-%s' % (str(r[0]), str(r[1][0]), str(r[1][1]), str(r[1][2]), str(r[1][3]))

                for i, w in enumerate(sent):
                    t = '%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s' % (w[0].encode('utf-8'), w[1].encode('utf-8'), i,
                                                            w[2].encode('utf-8'), w[3].encode('utf-8'),
                                                            w[4].encode('utf-8'), w[5].encode('utf-8'),
                                                            w[6].encode('utf-8'))
                    print >> f, t
                print >> f


def add_to_cluster(cluster, mention_pair_index, mention_pair_p, posit_i):
    """
    :param cluster: 1D: n_cluster; list, 2D: n_mentions; set
    :param mention_pair_index: int: index
    :param mention_pair_p: float: prob
    :param posit_i: 2D: n_pairs, 1D: (m_sent_i, m_span, a_sent_j, a_span)
    """

    if mention_pair_p < 0.5:
        return cluster

    posit = posit_i[mention_pair_index]
    ment = (posit[0], posit[1])
    ant = (posit[2], posit[3])

    for c in cluster:
        if ant in c:
            c.add(ment)
            break
    else:
        cluster.append(set([ant, ment]))

    return cluster


def output_results(fn, corpus, doc_names, clusters):
    with open(fn, 'w') as f:
        corefs = []

        for doc, cluster in zip(corpus, clusters):
            d_corefs = []

            for sent in doc:
                d_corefs.append([[] for w_i in xrange(len(sent))])
            corefs.append(d_corefs)

            cluster.sort()

            for c_i, c in enumerate(cluster):
                for m in c:
                    sent_i = m[0]
                    span = m[1]

                    for j in xrange(span[0], span[1]+1):
                        d_corefs[sent_i][j].append((span, c_i))

        for doc, doc_name, d_corefs in zip(corpus, doc_names, corefs):
            print >> f, doc_name

            for sent, s_corefs in zip(doc, d_corefs):
                for i, w in enumerate(sent):
                    coref = s_corefs[i]
                    coref.sort()
                    text = ''

                    for c in coref:
                        span = c[0]
                        span_len = span[1] - span[0]
                        c_id = c[1]

                        if span_len == 0:
                            if text.startswith('|'):  # |1)
                                text = '(%d)' % c_id + text
                            elif text.startswith('(') and text.endswith(')'):  # (1)
                                text += '|(%d)' % c_id
                            else:
                                if text.endswith(')'):  # 1)
                                    text = '(%d)|' % c_id + text
                                else:
                                    text += '(%d)' % c_id
                        else:
                            if span[0] == i:
                                if len(text) == 0:
                                    text = '(%d' % c_id
                                else:
                                    if text.endswith(')'):
                                        text = '(%d|' % c_id + text
                                    else:
                                        text += '|(%d' % c_id
                            elif span[1] == i:
                                if len(text) == 0:
                                    text += '%d)' % c_id
                                else:
                                    text += '|%d)' % c_id

                    if len(text) == 0:
                        text = '-'

                    print >> f, '%s\t%s\t%d\t%s\t%s' % (w[0].encode('utf-8'), w[1].encode('utf-8'), i, w[2].encode('utf-8'), text)
                print >> f
            print >> f, '#end document'
