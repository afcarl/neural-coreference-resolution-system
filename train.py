import sys
import time
import math
import random

from io_utils import Vocab, load_conll, load_init_emb
from utils import get_init_emb
from preprocess import get_gold_mentions, get_cand_mentions, check_coverage_of_cand_mentions, convert_words_into_ids, get_features, theano_format
from nn import Model
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
        print '\tVocabulary Size: %d' % vocab_word.size()

    """ Load corpora """
    print '\n\tLoading Corpora...'

    train_corpus, train_doc_names, vocab_word = load_conll(path=argv.train_data, vocab=vocab_word)
    if emb is None:
        emb = get_init_emb(vocab_word, argv.emb)
    print '\t\tTrain Documents: %d' % len(train_corpus)

    if argv.dev_data:
        dev_corpus, dev_doc_names, _ = load_conll(path=argv.dev_data, vocab=vocab_word)
        print '\t\tDev   Documents: %d' % len(dev_corpus)

    if argv.test_data:
        test_corpus, test_doc_names, _ = load_conll(path=argv.test_data, vocab=vocab_word)
        print '\t\tTest  Documents: %d' % len(test_corpus)

    """ Extract gold mentions: Train=155,560, Dev=19,156, Test=19,764 """
    # gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=(bos, eos)
    # gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=coref_id
    print '\n\tExtracting Gold Mentions...'
    print '\t\tTRAIN',
    tr_gold_ments, tr_gold_corefs = get_gold_mentions(train_corpus, check=argv.check)
    print '\t\tDEV  ',
    dev_gold_ments, dev_gold_corefs = get_gold_mentions(dev_corpus)

    """ Extract cand mentions """
    # cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    print '\n\tExtracting Cand Mentions...'
    print '\t\tTRAIN',
    tr_cand_ments = get_cand_mentions(train_corpus, check=argv.check)
    print '\t\tDEV  ',
    dev_cand_ments = get_cand_mentions(dev_corpus)

    """ Check the coverage: Coverage 95.0%, Rate 1:3.5 by Berkeley System """
    print '\n\tChecking the Coverage of the Candidate Mentions...'
    check_coverage_of_cand_mentions(tr_gold_ments, tr_cand_ments)
    check_coverage_of_cand_mentions(dev_gold_ments, dev_cand_ments)

    """ Convert words into IDs """
    print '\n\tConverting Words into IDs...'
    print '\t\tVocabulary Size: %d' % vocab_word.size()

    tr_word_ids = convert_words_into_ids(corpus=train_corpus[:10], vocab_word=vocab_word)
    dev_word_ids = convert_words_into_ids(corpus=dev_corpus[:10], vocab_word=vocab_word)

    if argv.test_data:
        test_word_ids = convert_words_into_ids(corpus=test_corpus, vocab_word=vocab_word)

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

    tr_phi  = get_features(tr_word_ids, tr_cand_ments, tr_gold_ments, tr_gold_corefs)
    dev_phi = get_features(dev_word_ids, dev_cand_ments, dev_gold_ments, dev_gold_corefs, True)

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

    """ Count the number of features """
    n_tr_phi_total = reduce(lambda a, b: a + reduce(lambda c, d: c + len(d), b, 0), tr_phi[-2], 0)
    n_tr_phi_t = reduce(lambda a, b: a + reduce(lambda c, d: c + np.sum(d), b, 0), tr_phi[-2], 0)
    n_tr_phi_f = n_tr_phi_total - n_tr_phi_t

    n_dev_phi_total = reduce(lambda a, b: a + reduce(lambda c, d: c + len(d), b, 0), dev_phi[-2], 0)
    n_dev_phi_t = reduce(lambda a, b: a + reduce(lambda c, d: c + np.sum(d), b, 0), dev_phi[-2], 0)
    n_dev_phi_f = n_dev_phi_total - n_dev_phi_t
    print '\t\tTrain Features Total: %d\tRate: P:N\t%d:%d' % (n_tr_phi_total, n_tr_phi_t, n_tr_phi_f)
    print '\t\tDev   Features Total: %d\tRate: P:N\t%d:%d' % (n_dev_phi_total, n_dev_phi_t, n_dev_phi_f)

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
            model.y     : tr_samples[4][bos: eos]
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
            model.y     : dev_samples[4][bos: eos]
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
        correct = 0.
        correct_t = 0.
        correct_f = 0.
        total = 0.
        total_r = 0.
        total_p = 0.

        for i, index in enumerate(indices):
            if i % 100 == 0 and i != 0:
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

        total_f = total - total_r
        accuracy = correct / total
        accuracy_t = correct_t / total_r
        accuracy_f = correct_f / total_f

        precision = correct_t / total_p
        recall = correct_t / total_r
        f = 2 * precision * recall / (precision + recall)

        print '\n\tNegative Log Likelihood: %f\tTime: %f seconds' % (total_loss, (end - start))
        print '\tAcc Total:     %f\tCorrect: %d\tTotal: %d' % (accuracy, correct, total)
        print '\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (accuracy_t, correct_t, total_r)
        print '\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (accuracy_f, correct_f, total_f)
        print '\tPrecision:     %f\tCorrect: %d\tTotal: %d' % (precision, correct_t, total_p)
        print '\tRecall:        %f\tCorrect: %d\tTotal: %d' % (recall, correct_t, total_r)
        print '\tF1:            %f' % f

        predict(epoch, dev_f, dev_corpus, dev_doc_names, dev_indices, dev_phi[-1])


def set_model(argv, vocab_word, init_emb):
    x_span = T.imatrix('x_span')
    x_word = T.imatrix('x_word')
    x_ctx  = T.imatrix('x_ctx')
    x_dist = T.ivector('x_dist')
    y      = T.ivector('y')

    """ Set params for the model """
    n_vocab    = vocab_word.size()
    dim_x_word = argv.emb
    dim_x_dist = 11  # (0, ..., 10-)
    dim_h      = argv.hidden
    L2_reg     = argv.reg

    """ Instantiate the model """
    return Model(x_span=x_span, x_word=x_word, x_ctx=x_ctx, x_dist=x_dist, y=y, init_emb=init_emb, n_vocab=n_vocab,
                 dim_w=dim_x_word, dim_d=dim_x_dist, dim_h=dim_h, L2_reg=L2_reg)
