import sys
import time
import math
import random

from io_utils import Vocab, load_conll, load_init_emb
from preprocess import get_gold_mentions, get_cand_mentions, check_coverage_of_cand_mentions, convert_words_into_ids, set_cand_ment_coref, theano_format
from feature_extractors import set_word_id_for_ment, get_features
from nn_emb import Model
from test import predict

import numpy as np
import theano
import theano.tensor as T


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
        outputs=[model.y_hat_index, model.p_y_hat,
                 model.correct, model.correct_t, model.correct_f, model.total_p, model.total_r],
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

    batch_size = argv.batch
    n_batches = n_tr_phi_total / batch_size
    indices = range(n_batches)

    print 'Training START\n'
    print 'Mini-Batch Samples: %d\n' % n_batches

    for epoch in xrange(argv.epoch):
        random.shuffle(indices)

        print '\nEpoch: %d' % (epoch + 1)
        print 'TRAIN'
        print '\tIndex: ',
        start = time.time()

        total_loss = 0.
        correct = np.zeros(9, dtype='float32')
        correct_t = np.zeros(9, dtype='float32')
        correct_f = np.zeros(9, dtype='float32')
        total = 0.
        total_r = np.zeros(9, dtype='float32')
        total_p = np.zeros(9, dtype='float32')

        for i, index in enumerate(indices):
            if i % 1000 == 0 and i != 0:
                print '%d' % i,
                sys.stdout.flush()

            loss, crr, crr_t, crr_f, ttl_p, ttl_r = train_f(index * batch_size, (index+1) * batch_size)

            assert not math.isnan(loss), 'Index: %d  Batch Index: %d' % (i, index)

            total_loss += loss
            correct += crr
            correct_t += crr_t
            correct_f += crr_f
            total += batch_size
            total_p += ttl_p
            total_r += ttl_r

        end = time.time()
        print '\n\tTime: %f seconds' % (end - start)
        show_results(total, total_p, total_r, correct, correct_t, correct_f, total_loss)

        predict(epoch, dev_f, dev_corpus, dev_doc_names, dev_indices, dev_posit)


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
                 init_emb=init_emb, n_vocab=n_vocab, dim_w=dim_x_word, dim_d=dim_x_dist, dim_h=dim_h, L2_reg=L2_reg)


def show_results(total, total_p, total_r, correct, correct_t, correct_f, total_loss=None):
    total_f = total - total_r
    accuracy = correct / total
    accuracy_t = correct_t / total_r
    accuracy_f = correct_f / total_f

    precision = correct_t / total_p
    recall = correct_t / total_r
    fs = 2 * precision * recall / (precision + recall)

    if total_loss is not None:
        print '\n\tNegative Log Likelihood: %f\n' % total_loss

    t = 0.1
    for crr, crr_t, crr_f, acc, acc_t, acc_f, ttl_f, ttl_p, ttl_r, pr, rc, f in zip(correct, correct_t, correct_f,
                                                                                    accuracy, accuracy_t, accuracy_f,
                                                                                    total_f, total_p, total_r,
                                                                                    precision, recall, fs):
        print '\n\tThreshold Prob:%f' % t
        print '\t\tAcc Binary:    %f\tCorrect: %d\tTotal: %d' % (acc, crr, total)
        print '\t\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (acc_t, crr_t, ttl_r)
        print '\t\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (acc_f, crr_f, ttl_f)
        print '\t\tPrecision:     %f\tCorrect: %d\tTotal: %d' % (pr, crr_t, ttl_p)
        print '\t\tRecall:        %f\tCorrect: %d\tTotal: %d' % (rc, crr_t, ttl_r)
        print '\t\tF1:            %f' % f

        t += 0.1

