import sys
import time
import math
import random

from io_utils import Vocab, load_conll, load_init_emb
from preprocess import get_gold_mentions, get_cand_mentions, check_coverage_of_cand_mentions, convert_words_into_ids, set_cand_ment_coref
from feature_extractors import set_word_id_for_ment, get_features
from nn_utils import sample_weights, L2_sqr, sigmoid, relu, tanh
from optimizers import sgd_w

import numpy as np
import theano
import theano.tensor as T

theano.config.floatX = 'float32'
np.random.seed(0)


class Model(object):
    def __init__(self, x_span, x_word, x_ctx, x_dist, x_slen, y, init_emb, n_vocab, dim_w_p, dim_d, dim_h, L2_reg):
        """
        :param x_span: 1D: batch, 2D: limit * 2 (10); elem=word id
        :param x_word: 1D: batch, 2D: 4 (m_first, m_last, a_first, a_last); elem=word id
        :param x_ctx : 1D: batch, 2D: window * 2 * 2 (20); elem=word id
        :param x_dist: 1D: batch; 2D: 2; elem=[sent dist, ment dist]
        :param x_slen: 1D: batch; 2D: 3; elem=[m_span_len, a_span_len, head_match]
        :param y     : 1D: batch
        """

        self.input  = [x_span, x_word, x_ctx, x_dist, x_slen, y]
        self.x_span = x_span
        self.x_word = x_word
        self.x_ctx  = x_ctx
        self.x_dist = x_dist
        self.x_slen = x_slen
        self.y      = y

        """ Dimensions """
        dim_w_a = dim_w_p / 5
        dim_x_a = dim_w_a * (5 + 2 + 2 + 1)
        dim_x_p = dim_w_p * (10 + 4 + 4 + 2 + 3) + dim_x_a
        batch = y.shape[0]

        """ Hyper Parameters for Cost Function """
        self.a1 = 0.5
        self.a2 = 1.2
        self.a3 = 1.

        """ Params """
        if init_emb is None:
            self.W_a_w = theano.shared(sample_weights(n_vocab, dim_w_a))
            self.W_p_w = theano.shared(sample_weights(n_vocab, dim_w_p))
        else:
            self.W_a_w = theano.shared(init_emb)
            self.W_p_w = theano.shared(init_emb)

        self.W_a_l = theano.shared(sample_weights(5, dim_w_a))
        self.W_a_o = theano.shared(sample_weights(dim_x_a, 1))

        self.W_p_d = theano.shared(sample_weights(dim_d, dim_w_p))
        self.W_p_l = theano.shared(sample_weights(7, dim_w_p))
        self.W_p_h = theano.shared(sample_weights(dim_x_p, dim_h))
        self.W_p_o = theano.shared(sample_weights(dim_h))

        self.params = [self.W_p_d, self.W_p_l, self.W_a_l, self.W_p_h, self.W_p_o, self.W_a_o]

        """ Anaphoric Layer """
        x_vec_a = T.concatenate([x_span[0][:x_span.shape[1]/2],
                                 x_word[0][:x_word.shape[1]/2],
                                 x_ctx[0][:x_ctx.shape[1]/2]])

        x_a_w = self.W_a_w[x_vec_a]       # 1D: batch, 2D: (limit * 1 + 2 + ctx), 3D: dim_w_a
        x_a_l = self.W_a_l[x_slen[0][0]]  # 1D: dim_w_a
        h_a = T.concatenate([x_a_w.flatten(), x_a_l])

        """ Pair Layer """
        x_p_w_in = T.concatenate([x_span, x_word, x_ctx], 1).flatten()  # 1D: batch * (limit * 2 + 4 + 20)
        x_p_w = self.W_p_w[x_p_w_in]  # 1D: batch, 2D: (limit * 2 + 4 + ctx * 2), 3D: dim_w
        x_p_l = self.W_p_l[x_slen]    # 1D: batch, 2D: 3, 3D: dim_w
        x_p_d = self.W_p_d[x_dist]    # 1D: batch, 2D: 2, 3D: dim_w
        h_p = T.concatenate([x_p_w.reshape((batch, -1)), x_p_d.reshape((batch, -1)), x_p_l.reshape((batch, -1))], 1)
        g_p = tanh(T.dot(T.concatenate([h_p, T.repeat(h_a.dimshuffle('x', 0), batch, 0)], 1), self.W_p_h))

        """ Output Layer """
        p_y_a = T.dot(h_a, self.W_a_o)  # p_y_a: 1D: 1; elem=scalar
        p_y_p = T.dot(g_p, self.W_p_o)  # p_y_p: 1D: batch
        p_y = T.concatenate([p_y_a, p_y_p])

        """ Label Set """
        y_0 = T.switch(T.sum(y), 0, 1)  # y_0: 1 if the mention is non-anaph else 0
        y_all = T.concatenate([y_0.dimshuffle('x'), y])

        """ Predicts """
        self.y_hat = T.argmax(p_y)
        self.p_y_hat = p_y[T.argmax(p_y - T.min(p_y) * y_all)]

        """ Cost Function """
        self.nll = T.max(self.miss_cost(T.arange(y_all.shape[0]), y_all) * (1 + p_y - self.p_y_hat))
        self.cost = self.nll + L2_reg * L2_sqr(params=self.params) / 2

        """ Optimization """
        self.updates = sgd_w(self.cost, self.params, self.W_p_w, x_p_w, self.W_a_w, x_a_w)

        """ Check Results """
        self.total_p = T.switch(self.y_hat, 1, 0)
        self.total_r = 1 - y_0
        self.correct = y_all[self.y_hat]
        self.correct_t = T.switch(self.correct, T.switch(y_0, 0, 1), 0)
        self.correct_f = T.switch(self.correct, T.switch(y_0, 1, 0), 0)

    def miss_cost(self, y_index, y):
        return T.switch(y_index,
                        T.switch(y[0], self.a1, self.a3 * (1 - y[y_index - 1])),
                        T.switch(y[0], 0, self.a2))


def main(argv):
    print '\nSYSTEM START'
    print '\nMODE: Training'

    ###################
    # PREPROCESS DATA #
    ###################

    """ Load initial embedding file """
    vocab_word = Vocab()
    emb = None
    if argv.init_emb:
        print '\n\tInitial Embedding Loading...'
        emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
        print '\t\tVocabulary Size: %d' % vocab_word.size()

    """ Load corpora """
    print '\n\tLoading Corpora...'
    tr_corpus, tr_doc_names, vocab_word = load_conll(path=argv.train_data, vocab=vocab_word, data_size=argv.data_size)
    dev_corpus, dev_doc_names, _ = load_conll(path=argv.dev_data, vocab=vocab_word, data_size=argv.data_size)
    print '\t\tTrain Documents: %d' % len(tr_corpus)
    print '\t\tDev   Documents: %d' % len(dev_corpus)

    """ Extract gold mentions CoNLL-2012: Train=155,560, Dev=19,156, Test=19,764 """
    # gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=(bos, eos)
    # gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=coref_id
    print '\n\tExtracting Gold Mentions...'
    print '\t\tTRAIN',
    tr_gold_ments = get_gold_mentions(tr_corpus, check=argv.check)
    print '\t\tDEV  ',
    dev_gold_ments = get_gold_mentions(dev_corpus)

    """ Extract cand mentions """
    # cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    print '\n\tExtracting Cand Mentions...'
    print '\t\tTRAIN',
    tr_cand_ments = get_cand_mentions(tr_corpus, check=argv.check)
    print '\t\tDEV  ',
    dev_cand_ments = get_cand_mentions(dev_corpus)

    """ Convert words into IDs """
    print '\n\tConverting Words into IDs...'
    print '\t\tVocabulary Size: %d' % vocab_word.size()

    tr_word_ids = convert_words_into_ids(corpus=tr_corpus, vocab_word=vocab_word)
    dev_word_ids = convert_words_into_ids(corpus=dev_corpus, vocab_word=vocab_word)

    """ Set word ids for mentions """
    tr_gold_ments = set_word_id_for_ment(tr_word_ids, tr_gold_ments)
    tr_cand_ments = set_word_id_for_ment(tr_word_ids, tr_cand_ments)
    dev_gold_ments = set_word_id_for_ment(dev_word_ids, dev_gold_ments)
    dev_cand_ments = set_word_id_for_ment(dev_word_ids, dev_cand_ments)

    """ Set coref ids for cand mentions """
    tr_cand_ments = set_cand_ment_coref(tr_gold_ments, tr_cand_ments)
    dev_cand_ments = set_cand_ment_coref(dev_gold_ments, dev_cand_ments)

    """ Check the coverage: Coverage 95.0%, Rate 1:3.5 by Berkeley System """
    print '\n\tChecking the Coverage of the Candidate Mentions...'
    check_coverage_of_cand_mentions(tr_gold_ments, tr_cand_ments)
    check_coverage_of_cand_mentions(dev_gold_ments, dev_cand_ments)

    """ Extract features """
    print '\n\tExtracting features...'

    """
    phi = (span, word, ctx, dist, label, position)
    span    : 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: limit * 2; elem=word id
    word    : 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: [m_first, m_last, a_first, a_last]; elem=word id
    ctx     : 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: window * 2 * 2; elem=word id
    dist    : 1D: n_doc, 2D: n_ments, 3D: n_cand_ants; elem=sent dist
    label   : 1D: n_doc, 2D: n_ments; elem=0/1
    position: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants; elem=(sent_m_i, span_m, sent_a_i, span_a)
    """

    tr_phi, tr_posit = get_features(tr_cand_ments, False, argv.n_cands)
    dev_phi, dev_posit = get_features(dev_cand_ments, True, argv.n_cands)

    """ Count the number of features """
    n_tr_phi_total = reduce(lambda a, b: a + reduce(lambda c, d: c + len(d), b, 0), tr_phi, 0)
    n_tr_phi_t = reduce(lambda a, b: a + reduce(lambda c, d: c + reduce(lambda e, f: e + np.sum(f[-1]), d, 0), b, 0), tr_phi, 0)
    n_tr_phi_f = n_tr_phi_total - n_tr_phi_t

    n_dev_phi_total = reduce(lambda a, b: a + reduce(lambda c, d: c + len(d), b, 0), dev_phi, 0)
    n_dev_phi_t = reduce(lambda a, b: a + reduce(lambda c, d: c + reduce(lambda e, f: e + np.sum(f[-1]), d, 0), b, 0), dev_phi, 0)
    n_dev_phi_f = n_dev_phi_total - n_dev_phi_t
    print '\t\tTrain Features Total: %d\tRate: P:N\t%d:%d' % (n_tr_phi_total, n_tr_phi_t, n_tr_phi_f)
    print '\t\tDev   Features Total: %d\tRate: P:N\t%d:%d' % (n_dev_phi_total, n_dev_phi_t, n_dev_phi_f)

    """ Convert into the Theano format """
    print '\n\tConverting features into the Theano Format...'

    """
    samples = (span, word, ctx, dist, label)
    span   : 1D: n_doc * n_ments * n_cand_ants, 2D: limit * 2; elem=word id
    word   : 1D: n_doc * n_ments * n_cand_ants, 2D: [m_first, m_last, a_first, a_last]; elem=word id
    ctx    : 1D: n_doc * n_ments * n_cand_ants, 2D: window * 2 * 2; elem=word id
    dist   : 1D: n_doc * n_ments * n_cand_ants; elem=sent dist
    label  : 1D: n_doc * n_ments * n_cand_ants; elem=0/1
    indices: 1D: n_doc * n_ments; elem=(bos, eos)
    """

    tr_samples, tr_indices = theano_format(tr_phi)
    dev_samples, dev_indices = theano_format(dev_phi)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '\nBuilding the model...'

    model = set_model(argv, vocab_word, emb)

    bos = T.iscalar('bos')
    eos = T.iscalar('eos')

    train_f = theano.function(
        inputs=[bos, eos],
        outputs=[model.nll, model.correct, model.correct_t, model.correct_f, model.total_p, model.total_r],
        updates=model.updates,
        givens={
            model.x_span: tr_samples[0][bos: eos],
            model.x_word: tr_samples[1][bos: eos],
            model.x_ctx : tr_samples[2][bos: eos],
            model.x_dist: tr_samples[3][bos: eos],
            model.x_slen: tr_samples[4][bos: eos],
            model.y     : tr_samples[5][bos: eos]
        },
        mode='FAST_RUN'
    )

    dev_f = theano.function(
        inputs=[bos, eos],
        outputs=[model.y_hat, model.correct, model.correct_t, model.correct_f, model.total_p, model.total_r],
        givens={
            model.x_span: dev_samples[0][bos: eos],
            model.x_word: dev_samples[1][bos: eos],
            model.x_ctx : dev_samples[2][bos: eos],
            model.x_dist: dev_samples[3][bos: eos],
            model.x_slen: dev_samples[4][bos: eos],
            model.y     : dev_samples[5][bos: eos]
        },
        mode='FAST_RUN'
    )

    ###############
    # TRAIN MODEL #
    ###############

    print 'Training START\n'
    print 'Docs: %d\n' % len(tr_indices)

    for epoch in xrange(argv.epoch):
        random.shuffle(tr_indices)

        print '\nEpoch: %d' % (epoch + 1)
        print 'TRAIN'
        print '\tIndex: ',
        start = time.time()

        total_loss = 0.
        correct = 0.
        correct_t = 0.
        correct_f = 0.
        total = 0.
        total_r = 0.
        total_p = 0.
        k = 0

        for m_indices in tr_indices:
            random.shuffle(m_indices)

            for index in m_indices:
                if k % 1000 == 0 and k != 0:
                    print '%d' % k,
                    sys.stdout.flush()

                loss, crr, crr_t, crr_f, ttl_p, ttl_r = train_f(index[0], index[1])
                assert not math.isnan(loss), 'Index: %d' % k

                total_loss += loss
                correct += crr
                correct_t += crr_t
                correct_f += crr_f
                total += 1
                total_p += ttl_p
                total_r += ttl_r
                k += 1

        end = time.time()
        print '\n\tTime: %f seconds' % (end - start)
        show_results(total, total_p, total_r, correct, correct_t, correct_f, total_loss)

        predict(epoch, dev_f, dev_corpus, dev_doc_names, dev_indices, dev_posit)


def predict(epoch, model, corpus, doc_names, indices, posits):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words, 4D: (doc_id, part_id, form, tag, syn, ne, coref_id)
    :param doc_names: 1D: n_doc; str
    :param indices: 1D: n_doc, 2D: n_ments; (bos, eos)=ment * all cand ants
    :param posits: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: (m_sent_i, m_span, a_sent_i, a_span)
    """

    assert len(indices) == len(posits)

    print '\nTEST'
    print '\tIndex: ',
    start = time.time()

    clusters = []

    correct = 0.
    correct_t = 0.
    correct_f = 0.
    total = 0.
    total_r = 0.
    total_p = 0.
    k = 0

    for d_indices, d_posits in zip(indices, posits):
        cluster = []

        for index, posit in zip(d_indices, d_posits):
            if k % 1000 == 0 and k != 0:
                print '%d' % k,
                sys.stdout.flush()

            y_hat, crr, crr_t, crr_f, ttl_p, ttl_r = model(index[0], index[1])

            correct += crr
            correct_t += crr_t
            correct_f += crr_f
            total += 1
            total_p += ttl_p
            total_r += ttl_r

            cluster = add_to_cluster(cluster, y_hat, posit)

            k += 1

        clusters.append(cluster)

    end = time.time()
    print '\n\tTime: %f seconds' % (end - start)
    show_results(total, total_p, total_r, correct, correct_t, correct_f)
    output_results(fn='result.epoch-%d.txt' % (epoch + 1), corpus=corpus, doc_names=doc_names, clusters=clusters)


def add_to_cluster(cluster, mention_pair_index, posit_i):
    """
    :param cluster: 1D: n_cluster; list, 2D: n_mentions; set
    :param mention_pair_index: int: index
    :param posit_i: 2D: n_pairs, 1D: (m_sent_i, m_span, a_sent_j, a_span)
    """

    if mention_pair_index == 0:
        return cluster

    posit = posit_i[mention_pair_index - 1]
    ment = (posit[0], posit[1])
    ant = (posit[2], posit[3])

    for c in cluster:
        if ant in c:
            c.add(ment)
            break
    else:
        cluster.append(set([ant, ment]))

    return cluster


def set_model(argv, vocab_word, init_emb):
    x_span = T.imatrix('x_span')
    x_word = T.imatrix('x_word')
    x_ctx  = T.imatrix('x_ctx')
    x_dist = T.imatrix('x_dist')
    x_slen = T.imatrix('x_slen')
    y      = T.ivector('y')

    """ Set params for the model """
    n_vocab    = vocab_word.size()
    dim_x_word = argv.emb
    dim_x_dist = 10  # (0, ..., 10-)
    dim_h      = argv.hidden
    L2_reg     = argv.reg

    """ Instantiate the model """
    return Model(x_span=x_span, x_word=x_word, x_ctx=x_ctx, x_dist=x_dist, x_slen=x_slen, y=y,
                 init_emb=init_emb, n_vocab=n_vocab, dim_w_p=dim_x_word, dim_d=dim_x_dist, dim_h=dim_h, L2_reg=L2_reg)


def show_results(total, total_p, total_r, correct, correct_t, correct_f, total_loss=None):
    total_f = total - total_r
    accuracy = correct / total
    accuracy_t = correct_t / total_r
    accuracy_f = correct_f / total_f

    precision = correct_t / total_p
    recall = correct_t / total_r
    f = 2 * precision * recall / (precision + recall)

    if total_loss is not None:
        print '\n\tNegative Log Likelihood: %f\n' % total_loss

    print '\t\tAcc Binary:    %f\tCorrect: %d\tTotal: %d' % (accuracy, correct, total)
    print '\t\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (accuracy_t, correct_t, total_r)
    print '\t\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (accuracy_f, correct_f, total_f)
    print '\t\tPrecision:     %f\tCorrect: %d\tTotal: %d' % (precision, correct_t, total_p)
    print '\t\tRecall:        %f\tCorrect: %d\tTotal: %d' % (recall, correct_t, total_r)
    print '\t\tF1:            %f' % f


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


def theano_format(samples):
    """
    samples = (span, word, ctx, dist, label, position)
    span: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: limit * 2; elem=word id
    word: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: [m_first, m_last, a_first, a_last]; elem=word id
    ctx: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants, 4D: window * 2 * 2; elem=word id
    dist: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants; elem=sent dist
    label: 1D: n_doc, 2D: n_ments; elem=0/1
    position: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants; elem=(sent_m_i, span_m, sent_a_i, span_a)
    """

    def shared(_sample):
        return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    sample_s = []
    sample_w = []
    sample_c = []
    sample_d = []
    sample_l = []
    sample_y = []

    indices = []

    for doc_samples in samples:
        d_indices = []
        for ment_samples in doc_samples:
            bos = len(sample_y)
            for sample in ment_samples:
                s, w, c, d, l, y = sample
                sample_s.append(s)
                sample_w.append(w)
                sample_c.append(c)
                sample_d.append(d)
                sample_l.append(l)
                sample_y.append(y)
            d_indices.append((bos, len(sample_y)))
        indices.append(d_indices)

    assert len(sample_s) == len(sample_w) == len(sample_c) == len(sample_d) == len(sample_l) == len(sample_y)
    return [shared(sample_s), shared(sample_w), shared(sample_c), shared(sample_d), shared(sample_l), shared(sample_y)], indices


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Coreference Resolution System')

    parser.add_argument('--mode', default='train', help='train/test')
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to dev data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--check', default=False, help='print each steps')

    """ NN architecture """
    parser.add_argument('--unit', default='lstm', help='Unit')
    parser.add_argument('--vocab', type=int, default=100000000, help='vocabulary size')
    parser.add_argument('--emb', type=int, default=50, help='dimension of embeddings')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--hidden', type=int, default=32, help='dimension of hidden layer')
    parser.add_argument('--layer', type=int, default=1, help='number of layers')

    """ training options """
    parser.add_argument('--save', type=bool, default=False, help='parameters to be saved or not')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.001, help='L2 Reg rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--no-shuffle', action='store_true', default=False, help='don\'t shuffle training data')
    parser.add_argument('--n_cands', type=int, default=10, help='number of ant candidates')
    parser.add_argument('--data_size', type=int, default=10000, help='number of docs')

    """ test options """
    parser.add_argument('--model', default=None, help='path to model')
    parser.add_argument('--arg_dict', default=None, help='path to arg dict')
    parser.add_argument('--vocab_dict', default=None, help='path to vocab dict')
    parser.add_argument('--emb_dict', default=None, help='path to emb dict')

    argv = parser.parse_args()
    main(argv)
