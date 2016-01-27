__author__ = 'hiroki'


import re

from io import UNK
from utils import shuffle
from nn_utils import sample_weights, get_zeros

import numpy as np

RE_PH = re.compile(ur'[A-Z]+')


def get_init_emb(vocab_word, emb_dim):
    emb = sample_weights(size_x=vocab_word.size(), size_y=emb_dim)
    emb[0] = get_zeros((1, emb_dim))
    return np.asarray(emb)


def extract_gold_mentions(corpus):
    gold = []
    count = 0

    for doc in corpus:
        g_doc = []
        for i, sent in enumerate(doc):
            g_sent = extract_mentions(i, sent)
            g_doc.append(g_sent)
            count += len(g_sent)
        gold.append(g_doc)

    print 'Gold Mentions: %d' % count

    return gold


def extract_mentions(index, sent):
    mentions = []
    prev = []

    for i, w in enumerate(sent):
        coref = w[6].split('|')

        for m in coref:
            if m.startswith('('):
                if m.endswith(')'):
                    m_id = int(m[1:-1])
                    mentions.append((index, i, i, m_id))
                else:
                    m_id = int(m[1:])
                    prev.append((index, i, i, m_id))
            else:
                if m.endswith(')'):
                    m_id = int(m[:-1])

                    for j, p in enumerate(prev):
                        if m_id == p[-1]:
                            mentions.append((p[0], p[1], i, p[-1]))
                            prev.pop(j)
                            break
                    else:
                        print 'Error at extract_mentions(): %s' % str(sent)
                        exit()

    assert len(prev) == 0

    return mentions


def extract_mentions_bio(sent):
    mentions = []
    prev = []

    for w in sent:
        mention = []
        coref = w[6].split('|')

        for m in coref:
            if m.startswith('('):
                if m.endswith(')'):
                    m_id = int(m[1:-1])
                else:
                    m_id = int(m[1:])
                    prev.append(m_id)

                label = (0, m_id)
                mention.append(label)
            else:
                if m.endswith(')'):
                    m_id = int(m[:-1])

                    if m_id in prev:
                        label = (1, m_id)
                        mention.append(label)
                        prev.remove(m_id)
                else:
                    if prev:
                        for m_id in prev:
                            label = (1, m_id)
                            mention.append(label)
                    else:
                        label = (2, -1)
                        mention.append(label)

        mentions.append(mention)

    return mentions    


def extract_cand_mentions(corpus):
    cand = []
    count = 0

    for doc in corpus:
        mentions = []
        for i, sent in enumerate(doc):
            mention = []

            """ Extracting NP, Pro-Nom, NE mentions """
            mention.extend(extract_np(i, sent))
            mention.extend(extract_pronominals(i, sent))
            mention.extend(extract_ne(i, sent))

            """ Removing duplicates, and sorting """
            mention = list(set(mention))
            mention.sort()
            mentions.append(mention)

            count += len(mention)
        cand.append(mentions)

    print 'Cand Mentions: %d' % count
    return cand


def extract_np(index, sent):
    br_l = '('
    br_r = ')'
    bop = []  # beginning of a phrase
    phrases = []

    for i, w in enumerate(sent):
        syn = w[4]
        n_phrases_l = syn.count(br_l)
        n_phrases_r = syn.count(br_r)
        non_terminals = RE_PH.findall(syn)

        if n_phrases_l > 0:
            for c in non_terminals:
                bop.append((c, i))

        for j in xrange(n_phrases_r):
            c = bop.pop()
            phrases.append((c[0], c[1], i))

    return [(index, p[1], p[2]) for p in phrases if p[0] == 'NP']


def extract_pronominals(index, sent):
    return [(index, i, i) for i, w in enumerate(sent) if w[3] in ['PRP', 'PRP$']]
"""
    pros = ['PRP', 'PRP$']
    mentions = []

    for i, w in enumerate(sent):
        pos = w[3]

        if pos in pros:
            mentions.append((index, i, i))

    return mentions
"""

def extract_ne(index, sent):
    begin = -1
    except_nes = ['CARDINAL', 'QUANTITY', 'PERCENT']
    mentions = []

    for i, w in enumerate(sent):
        ne = w[5]

        if ne.startswith('(') and ne[1:-1] not in except_nes:
            begin = i

            if ne.endswith(')'):
                begin = -1
                mentions.append((index, i, i))
        elif ne.endswith(')') and begin > -1:
            mentions.append((index, begin, i))
            begin = -1

    return mentions


def check_coverage_of_cand_mentions(gold, cand):
    assert len(gold) == len(cand)

    t_count = 0
    g_total = 0
    c_total = 0

    for g_doc, c_doc in zip(gold, cand):

        assert len(g_doc) == len(c_doc)

        for g_sent, c_sent in zip(g_doc, c_doc):
            c_sent = [(c[1], c[2]) for c in c_sent]

            for g in g_sent:
                if (g[1], g[2]) in c_sent:
                    t_count += 1

            g_total += len(g_sent)
            c_total += len(c_sent)

    f_count = c_total - t_count

    print '\t\tCoverage: %f' % (t_count / float(g_total))
    print '\t\tCandidate True-False Rate: %f:%f' % (t_count / float(c_total), f_count / float(c_total))


def convert_words_into_ids(corpus, vocab_word):
    id_corpus = []

    for doc in corpus:
        id_doc = []
        for sent in doc:
            w_ids = []

            for w in sent:  # w=(doc_id, part_id, word, POS, syn, ne, coref)
                w_id = vocab_word.get_id(w[2])

                """ID for unknown word"""
                if w_id is None:
                    w_id = vocab_word.get_id(UNK)

                w_ids.append(w_id)

            assert len(sent) == len(w_ids)
            id_doc.append(w_ids)

        assert len(doc) == len(id_doc)
        id_corpus.append(id_doc)

    assert len(corpus) == len(id_corpus)

    return id_corpus


def extract_features(id_corpus, gold_mentions, cand_mentions, emb):
    """
    :param id_corpus: 1D: n_doc, 2D; n_sents, 3D: n_words; elem=word_id
    :param gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_cand_mentions; elem=(sent_index, i,j, coref_id)
    :param cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_cand_mentions; elem=(sent_index, i,j)
    :return: features: 1D: n_doc, 2D: n_sents, 3D: n_words * 2
    """

    def get_head_word_id(sent_index, head_index):
        return doc[sent_index][head_index]

    def get_antecedent(coref_id, antecedents):
        """
        :return: the closest antecedent
        """
        antecedents.reverse()
        for a in antecedents:
            if coref_id == a[-1]:
                return [a]
        return []

    def get_negative_sample(ment, ant):
        m_sent_id = ment[0]
        a_sent_id = ant[0]
        assert a_sent_id <= m_sent_id

        for cands in c_mentions[a_sent_id:m_sent_id+1]:
            for c in cands:
                if ant[1] < c[1]:
                    return [c]
        return []

    p_features = []
    n_features = []

    for i in xrange(len(id_corpus)):
        doc = id_corpus[i]
        g_mentions = []

        for mention in gold_mentions[i]:
            g_mentions.extend(mention)
        g_mentions.sort()

        c_mentions = cand_mentions[i]

        for j, m in enumerate(g_mentions):
            m_head_id = get_head_word_id(sent_index=m[0], head_index=m[2])
            m_phi = emb[m_head_id]

            gold_ants = get_antecedent(coref_id=m[-1], antecedents=g_mentions[:j])

            for gold_a in gold_ants:
                a_head_id = get_head_word_id(sent_index=gold_a[0], head_index=gold_a[2])
                a_phi = emb[a_head_id]
                p_features.append(np.concatenate((m_phi, a_phi)))

                neg_cand_a = get_negative_sample(ment=m, ant=gold_a)

                for neg_a in neg_cand_a:
                    n_head_id = get_head_word_id(sent_index=neg_a[0], head_index=neg_a[2])
                    n_phi = emb[n_head_id]

                    n_features.append(np.concatenate((m_phi, n_phi)))

    return p_features, n_features


def convert_into_theano_input_format(p_phi, n_phi):
    sample_x = np.concatenate((p_phi, n_phi), axis=0)
    sample_y = np.concatenate((np.ones((len(p_phi))), np.zeros((len(n_phi)))), axis=0)

    assert len(sample_x) == len(sample_y)

    return shuffle(sample_x, sample_y)
