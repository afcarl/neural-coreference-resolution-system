import sys
import time
import math

from io_utils import Vocab, load_conll, load_init_emb
from utils import get_init_emb, shuffle
from preprocess import get_gold_mentions, get_cand_mentions, check_coverage_of_cand_mentions, convert_words_into_ids, get_features, convert_into_theano_input_format
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

    train_corpus, vocab_word = load_conll(path=argv.train_data, vocab=vocab_word)
    if emb is None:
        emb = get_init_emb(vocab_word, argv.emb)
    print '\tTrain Documents: %d' % len(train_corpus)

    if argv.dev_data:
        dev_corpus, _ = load_conll(path=argv.dev_data, vocab=vocab_word)
        print '\tDev   Documents: %d' % len(dev_corpus)

    if argv.test_data:
        test_corpus, _ = load_conll(path=argv.test_data, vocab=vocab_word)
        print '\tTest  Documents: %d' % len(test_corpus)

    """ Extract gold mentions: Train=155,560, Dev=19,156, Test=19,764 """
    # gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=(bos, eos)
    # gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=coref_id
    print '\n\tExtracting Gold Mentions...'
    print '\t\tTRAIN',
    train_gold_mentions, train_gold_corefs = get_gold_mentions(train_corpus, check=argv.check)
    print '\t\tDEV  ',
    dev_gold_mentions, dev_gold_corefs = get_gold_mentions(dev_corpus)

    """ Extract cand mentions """
    # cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    print '\n\tExtracting Cand Mentions...'
    print '\t\tTRAIN',
    train_cand_mentions = get_cand_mentions(train_corpus, check=argv.check)
    print '\t\tDEV  ',
    dev_cand_mentions = get_cand_mentions(dev_corpus)

    """ Check the coverage: Coverage 95.0%, Rate 1:3.5 by Berkeley System """
    print '\n\tChecking the Coverage of the Candidate Mentions...'
    check_coverage_of_cand_mentions(train_gold_mentions, train_cand_mentions)
    check_coverage_of_cand_mentions(dev_gold_mentions, dev_cand_mentions)

    """ Convert words into IDs """
    print '\n\tConverting Words into IDs...'
    print '\tVocabulary Size: %d' % vocab_word.size()

    train_word_ids = convert_words_into_ids(corpus=train_corpus, vocab_word=vocab_word)
    dev_word_ids = convert_words_into_ids(corpus=dev_corpus, vocab_word=vocab_word)

    if argv.test_data:
        test_word_ids = convert_words_into_ids(corpus=test_corpus, vocab_word=vocab_word)

    """ Extract features """
    print '\n\tExtracting features'
    tr_x, tr_y, tr_p = get_features(train_word_ids, train_cand_mentions, train_gold_mentions, train_gold_corefs)
    dev_x, dev_y, dev_p = get_features(dev_word_ids, dev_cand_mentions, dev_gold_mentions, dev_gold_corefs, True)

    tr_sample_x, tr_sample_y = convert_into_theano_input_format(tr_x, tr_y)
    dev_sample_x, dev_sample_y = convert_into_theano_input_format(dev_x, dev_y)

    n_tr_phi_total = reduce(lambda a, b: a + reduce(lambda c, d: c + len(d), b, 0), tr_sample_y, 0)
    n_tr_phi_t = reduce(lambda a, b: a + reduce(lambda c, d: c + np.sum(d), b, 0), tr_sample_y, 0)
    n_tr_phi_f = n_tr_phi_total - n_tr_phi_t

    n_dev_phi_total = reduce(lambda a, b: a + reduce(lambda c, d: c + len(d), b, 0), dev_sample_y, 0)
    n_dev_phi_t = reduce(lambda a, b: a + reduce(lambda c, d: c + np.sum(d), b, 0), dev_sample_y, 0)
    n_dev_phi_f = n_dev_phi_total - n_dev_phi_t
    print '\tTrain Features Total: %d\tRate: P:N\t%d:%d' % (n_tr_phi_total, n_tr_phi_t, n_tr_phi_f)
    print '\tTest  Features Total: %d\tRate: P:N\t%d:%d' % (n_dev_phi_total, n_dev_phi_t, n_dev_phi_f)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '\nBuilding the model...'

    """ Allocate symbolic variables """
    x = T.imatrix('x')
    y = T.ivector('y')

    """ Set params for the model """
    n_vocab = vocab_word.size()
    dim_x = argv.emb
    dim_h = argv.hidden
    L2_reg = argv.reg
    batch = argv.batch

    """ Build the model """
    classifier = Model(x=x, y=y, init_emb=emb, n_vocab=n_vocab, dim_x=dim_x, dim_h=dim_h, n_label=1, L2_reg=L2_reg)

    train_model = theano.function(
        inputs=[x, y],
        outputs=[classifier.nll, classifier.correct],
        updates=classifier.updates,
        mode='FAST_RUN'
    )

    test_model = theano.function(
        inputs=[x, y],
        outputs=[classifier.correct, classifier.y_hat_index, classifier.y_hat_p],
        mode='FAST_RUN'
    )

    ###############
    # TRAIN MODEL #
    ###############

    n_samples = len(tr_sample_x)
    print 'Training START\n'

    for epoch in xrange(argv.epoch):
        tr_sample_x, tr_sample_y = shuffle(tr_sample_x, tr_sample_y)

        print '\nEpoch: %d' % (epoch + 1)
        print 'TRAIN'
        print '\tIndex: ',
        start = time.time()

        loss = 0.
        correct = 0.
        correct_t = 0.
        correct_f = 0.
        total = 0.
        total_t = 0.
        k = 0

        for doc_index in xrange(n_samples):
            d_sample_x = tr_sample_x[doc_index]
            d_sample_y = tr_sample_y[doc_index]

            for m_index in xrange(len(d_sample_x)):
                if k % 1000 == 0 and k != 0:
                    print '%d' % k,
                    sys.stdout.flush()

                _sample_x = d_sample_x[m_index]
                _sample_y = d_sample_y[m_index]

                l, c = train_model(_sample_x, _sample_y)

                if math.isnan(l):
                    print 'Doc Index: %d, Mention Index: %d' % (doc_index, m_index)
                    exit()

                loss += l
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
        avg_loss = loss / k

        total_f = total - total_t
        accuracy = correct / total
        accuracy_t = correct_t / total_t
        accuracy_f = correct_f / total_f

        print '\n\tAverage Negative Log Likelihood: %f\tTime: %f seconds' % (avg_loss, (end - start))
        print '\tAcc Total:     %f\tCorrect: %d\tTotal: %d' % (accuracy, correct, total)
        print '\tAcc Anaph:     %f\tCorrect: %d\tTotal: %d' % (accuracy_t, correct_t, total_t)
        print '\tAcc Non-Anaph: %f\tCorrect: %d\tTotal: %d' % (accuracy_f, correct_f, total_f)

        predict(epoch, test_model, dev_corpus, dev_sample_x, dev_sample_y, dev_p)
