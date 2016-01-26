__author__ = 'hiroki'

from io import Vocab, load_conll, load_init_emb
from preprocess import extract_gold_mentions, extract_cand_mentions, check_coverage_of_cand_mentions, convert_words_into_ids, extract_features


def main(argv):
    print '\nSYSTEM START'
    print '\nMODE: Training'

    """ Loading initial embedding file """
    vocab_word = Vocab()
    if argv.init_emb:
        print '\n\tInitial Embedding Loading...'
        init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
        print '\tVocabulary Size: %d' % vocab_word.size()

    """ Loading corpora """
    print '\n\tLoading Corpora...'

    train_corpus, vocab_word = load_conll(path=argv.train_data, vocab_word=vocab_word)
    print '\tTrain Documents: %d' % len(train_corpus)

    if argv.dev_data:
        dev_corpus, _ = load_conll(path=argv.dev_data, vocab_word=vocab_word)
        print '\tDev   Documents: %d' % len(dev_corpus)

    if argv.test_data:
        test_corpus, _ = load_conll(path=argv.test_data, vocab_word=vocab_word)
        print '\tTest  Documents: %d' % len(test_corpus)

    """ Extracting gold mentions: Train=155,560, Dev=19,156, Test=19,764 """
    print '\n\tExtracting Gold Mentions...'
    print '\t\tTRAIN',
    train_mentions = extract_gold_mentions(train_corpus)

    """ Extracting cand mentions """
    print '\n\tExtracting Cand Mentions...'
    print '\t\tTRAIN',
    train_cand_mentions = extract_cand_mentions(train_corpus)

    """ Checking the coverage: Coverage 95.0%, Rate 1:3.5 by Berkeley System """
    print '\n\tChecking the Coverage of the Candidate Mentions...'
    check_coverage_of_cand_mentions(train_mentions, train_cand_mentions)

    """ Converting words into IDs """
    print '\n\tConverting Words into IDs...'
    print '\tVocabulary Size: %d' % vocab_word.size()

    train_word_ids = convert_words_into_ids(corpus=train_corpus, vocab_word=vocab_word)

    if argv.dev_data:
        dev_word_ids = convert_words_into_ids(corpus=dev_corpus, vocab_word=vocab_word)

    if argv.test_data:
        test_word_ids = convert_words_into_ids(corpus=test_corpus, vocab_word=vocab_word)

    """ Extracting features """
    f1, f2 = extract_features(train_word_ids, train_mentions, train_cand_mentions)


