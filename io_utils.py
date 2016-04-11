import re
from collections import defaultdict

import numpy as np
import theano

PAD = u'PADDING'
UNK = u'UNKNOWN'
RE_NUM = re.compile(ur'[0-9]')


class Vocab(object):
    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def has_key(self, word):
        return self.w2i.has_key(word)

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab


def load_conll(path, vocab, vocab_size=None, data_size=10000, file_encoding='utf-8'):
    corpus = []
    doc_names = []
    word_freqs = defaultdict(int)

    """ Checking whether word IDs should be registered or not """
    register = False
    if vocab.size() == 0:
        vocab.add_word(PAD)
        vocab.add_word(UNK)
        register = True

    with open(path) as f:
        doc = []
        sent = []

        for line in f:
            es = line.rstrip().split()

            if line.startswith('#begin'):
                doc = []
                doc_names.append(" ".join(es))
            elif line.startswith('#end'):
                corpus.append(doc)
                if len(corpus) > data_size:
                    break
            elif len(es) > 10:
                doc_id = es[0].decode(file_encoding)
                part = es[1].decode(file_encoding)
                word = es[3].decode(file_encoding).lower()
                word = RE_NUM.sub(u'0', word)
                tag = es[4].decode(file_encoding)
                syn = es[5].decode(file_encoding)
                ne = es[10].decode(file_encoding)
                coref = es[-1].decode(file_encoding)

                sent.append((doc_id, part, word, tag, syn, ne, coref))
                word_freqs[word] += 1
            else:
                doc.append(sent)
                sent = []

    """ Registering word IDs """
    if register:
        for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
            if vocab_size is None or vocab.size() < vocab_size:
                vocab.add_word(w)
            else:
                break

    assert len(corpus) == len(doc_names)
    return corpus, doc_names, vocab


def load_init_emb(init_emb):
    vocab = Vocab()
    vocab.add_word(PAD)
    vocab.add_word(UNK)

    vec = {}
    with open(init_emb) as f_words:
        for line in f_words:
            line = line.strip().decode('utf-8').split()
            w = line[0]

            if w[1:-1] == UNK:
                w = UNK

            e = line[1:]
            if len(e) == 50 or len(e) == 300:
                vocab.add_word(w)
                vec[vocab.get_id(w)] = np.asarray(e, dtype=theano.config.floatX)

    dim = len(line[1:])

    vec[vocab.get_id(PAD)] = np.zeros(dim, dtype=theano.config.floatX)
    vec[vocab.get_id(UNK)] = np.zeros(dim, dtype=theano.config.floatX)

    emb = [[] for i in xrange(vocab.size())]
    for k, v in vec.items():
        emb[k] = v

    if vec.get(UNK) is None:
        """ averaging """
        avg = np.zeros(dim, dtype=theano.config.floatX)
        for i in xrange(len(emb)):
            avg += emb[i]
        avg_vec = avg / len(emb)
        emb[vocab.get_id(UNK)] = avg_vec

    emb = np.asarray(emb, dtype=theano.config.floatX)

    assert emb.shape[0] == vocab.size()
    return emb, vocab
