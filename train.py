__author__ = 'hiroki'

import sys
import time

from io import Vocab, load_conll, load_init_emb
from utils import shuffle
from preprocess import extract_gold_mentions, extract_cand_mentions, check_coverage_of_cand_mentions, convert_words_into_ids, extract_features, get_init_emb, convert_into_theano_input_format
from nn import Model

import numpy as np
import theano
import theano.tensor as T


def main(argv):
    print '\nSYSTEM START'
    print '\nMODE: Training'

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
    print '\n\tExtracting Gold Mentions...'
    print '\t\tTRAIN',
    train_mentions = extract_gold_mentions(train_corpus)

    """ Extract cand mentions """
    print '\n\tExtracting Cand Mentions...'
    print '\t\tTRAIN',
    train_cand_mentions = extract_cand_mentions(train_corpus)

    """ Check the coverage: Coverage 95.0%, Rate 1:3.5 by Berkeley System """
    print '\n\tChecking the Coverage of the Candidate Mentions...'
    check_coverage_of_cand_mentions(train_mentions, train_cand_mentions)

    """ Convert words into IDs """
    print '\n\tConverting Words into IDs...'
    print '\tVocabulary Size: %d' % vocab_word.size()

    train_word_ids = convert_words_into_ids(corpus=train_corpus, vocab_word=vocab_word)

    if argv.dev_data:
        dev_word_ids = convert_words_into_ids(corpus=dev_corpus, vocab_word=vocab_word)

    if argv.test_data:
        test_word_ids = convert_words_into_ids(corpus=test_corpus, vocab_word=vocab_word)

    """ Extract features """
    print '\n\tExtracting features'
    pos_phi, neg_phi = extract_features(train_word_ids, train_mentions, train_cand_mentions, emb)
    print '\tTrain Features P:N\t%d:%d' % (len(pos_phi), len(neg_phi))

    """ Convert features into theano input format """
    sample_x, sample_y = convert_into_theano_input_format(pos_phi, neg_phi)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '\nBuilding the model...'

    """ Allocate symbolic variables """
    x = T.fmatrix('x')
    y = T.fvector('y')

    """ Set params for the model """
    dim_x = len(sample_x[0])
    dim_h = argv.hidden
    L2_reg = argv.reg
    batch = argv.batch
    n_batch_samples = len(sample_x) / batch + 1

    """ Build the model """
    classifier = Model(x=x, y=y, dim_x=dim_x, dim_h=dim_h, n_label=1, L2_reg=L2_reg)

    train_model = theano.function(
        inputs=[x, y],
        outputs=classifier.nll,
        updates=classifier.updates
    )

    ###############
    # TRAIN MODEL #
    ###############
    print 'Training the model...\n'

    for epoch in xrange(argv.epoch):
        sample_x, sample_y = shuffle(sample_x, sample_y)

        print '\nEpoch: %d' % (epoch + 1)
        print '\tIndex: ',
        start = time.time()

        losses = []

        for b_index in xrange(n_batch_samples):
            if b_index % 100 == 0 and b_index != 0:
                print '%d' % b_index,
                sys.stdout.flush()

            _sample_x = sample_x[b_index * batch: (b_index+1) * batch]
            _sample_y = sample_y[b_index * batch: (b_index+1) * batch]

            if len(_sample_x) == 0:
                continue

            loss = train_model(_sample_x, _sample_y)
            losses.append(loss)

        end = time.time()
        avg_loss = np.mean(losses)

        print '\tTime: %f seconds' % (end - start)
        print '\tAverage Negative Log Likelihood: %f' % avg_loss

