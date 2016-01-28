__author__ = 'hiroki'


import re

from io import UNK
from utils import shuffle

import numpy as np

RE_PH = re.compile(ur'[A-Z]+')
NON_ANA = '-'


def get_gold_mentions(corpus):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref)
    :return: gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=(bos, eos)
    :return: gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=coref_id
    """

    gold_mentions = []
    gold_corefs = []
    count = 0

    for doc in corpus:
        doc_mention_spans = []
        doc_corefs = []

        for sent in doc:
            mention_spans, coref_ids = get_mention_spans(sent)
            doc_mention_spans.append(mention_spans)
            doc_corefs.append(coref_ids)
            count += len(mention_spans)

        assert len(doc_mention_spans) == len(doc_corefs)
        gold_mentions.append(doc_mention_spans)
        gold_corefs.append(doc_corefs)

    assert len(gold_mentions) == len(gold_corefs)
    print 'Gold Mentions: %d' % count
    return gold_mentions, gold_corefs


def get_mention_spans(sent):
    """
    :param sent: 1D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref)
    :return: mention_spans: 1D: n_mentions: elem=(bos, eos)
    :return: coref_ids: 1D: n_mentions: elem=coref_id
    """

    mention_spans = []
    coref_ids = []
    prev = []

    for i, w in enumerate(sent):
        mentions = w[6].split('|')

        for mention in mentions:
            if mention.startswith('('):
                if mention.endswith(')'):
                    coref_id = int(mention[1:-1])
                    mention_spans.append((i, i))
                    coref_ids.append(coref_id)
                else:
                    coref_id = int(mention[1:])
                    prev.append(((i, i), coref_id))
            else:
                if mention.endswith(')'):
                    coref_id = int(mention[:-1])

                    for j, p in enumerate(prev):
                        if coref_id == p[1]:
                            mention_spans.append((p[0][0], i))
                            coref_ids.append(coref_id)
                            prev.pop(j)
                            break
                    else:
                        print 'Error at extract_mentions(): %s' % str(sent)
                        exit()

    assert len(prev) == 0
    return mention_spans, coref_ids


def get_mentions_bio(sent):
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


def get_cand_mentions(corpus):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref_id)
    :return: cand: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    """
    cand_mentions = []
    count = 0

    for doc in corpus:
        mentions = []

        for i, sent in enumerate(doc):
            mention = []

            """ Extracting NP, Pro-Nom, NE mentions """
            mention.extend(get_np(sent))
            mention.extend(get_pronominals(sent))
            mention.extend(get_ne(sent))

            """ Removing duplicates, and sorting """
            mention = list(set(mention))
#            mention.sort()
            mentions.append(mention)
            count += len(mention)

        cand_mentions.append(mentions)

    print 'Cand Mentions: %d' % count
    return cand_mentions


def get_np(sent):
    br_l = '('
    br_r = ')'
    tmp_spans = []
    spans = []

    for i, w in enumerate(sent):
        syn = w[4]

        n_bos = syn.count(br_l)  # bos=beginning of span
        n_eos = syn.count(br_r)  # eos=end of span
        non_terminals = RE_PH.findall(syn)

        if n_bos > 0:
            for non_term_symbol in non_terminals:
                tmp_spans.append((non_term_symbol, i))

        for j in xrange(n_eos):
            non_term_symbol, bos = tmp_spans.pop()
            spans.append((non_term_symbol, bos, i))

    return [(bos, eos) for symbol, bos, eos in spans if symbol == 'NP']


def get_pronominals(sent):
    return [(i, i) for i, w in enumerate(sent) if w[3] in ['PRP', 'PRP$']]


def get_ne(sent):
    begin = -1
    except_nes = ['CARDINAL', 'QUANTITY', 'PERCENT']
    mentions = []

    for i, w in enumerate(sent):
        ne = w[5]

        if ne.startswith('(') and ne[1:-1] not in except_nes:
            begin = i

            if ne.endswith(')'):
                begin = -1
                mentions.append((i, i))
        elif ne.endswith(')') and begin > -1:
            mentions.append((begin, i))
            begin = -1

    return mentions


def check_coverage_of_cand_mentions(gold, cand):
    """
    :param gold: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    :param cand: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    """

    assert len(gold) == len(cand)

    t_count = 0
    g_total = 0
    c_total = 0

    for g_sents, c_sents in zip(gold, cand):

        assert len(g_sents) == len(c_sents)

        for g_sent, c_sent in zip(g_sents, c_sents):
            for g_span in g_sent:
                if g_span in c_sent:
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


def get_features(id_corpus, gold_mentions, cand_mentions, gold_corefs, emb):
    """
    :param id_corpus: 1D: n_doc, 2D; n_sents, 3D: n_words; elem=word_id
    :param gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    :param cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
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
        a_bos = ant[1][1]
        assert a_sent_id <= m_sent_id

        for cands in c_mentions[a_sent_id:m_sent_id+1]:
            for c in cands:
                if a_bos < c[1][1]:
                    return [c]
        return []

    p_features = []
    n_features = []

    for i, doc in enumerate(id_corpus):
        """ Convert sent level into document level, and remove [] """
        g_mentions = []
        for s_i, sent in enumerate(zip(gold_mentions[i], gold_corefs[i])):
            if sent[0]:
                for span, coref in zip(*sent):
                    g_mentions.append((s_i, span, coref))
        g_mentions.sort()

        c_mentions = [[(s_i, span) for span in sent] for s_i, sent in enumerate(cand_mentions[i]) if sent]
        c_mentions.sort()

        for j, mention in enumerate(g_mentions):
            m_head_id = get_head_word_id(sent_index=mention[0], head_index=mention[1][1])
            m_phi = emb[m_head_id]

            gold_ants = get_antecedent(coref_id=mention[-1], antecedents=g_mentions[:j])

            for gold_a in gold_ants:
                a_head_id = get_head_word_id(sent_index=gold_a[0], head_index=gold_a[1][1])
                a_phi = emb[a_head_id]
                p_features.append(np.concatenate((m_phi, a_phi)))

                neg_cand_a = get_negative_sample(ment=mention, ant=gold_a)

                for neg_a in neg_cand_a:
                    n_head_id = get_head_word_id(sent_index=neg_a[0], head_index=neg_a[1][1])
                    n_phi = emb[n_head_id]

                    n_features.append(np.concatenate((m_phi, n_phi)))

    return p_features, n_features


def get_test_features(id_corpus, cand_mentions, emb):
    """
    :param id_corpus: 1D: n_doc, 2D; n_sents, 3D: n_words; elem=word_id
    :param cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos,eos)
    :return: features: 1D: n_sents * n_cand_mentions, 2D: word_dim * 2
    :return: mention_indices: 1D: n_sents * n_cand_mentions; elem=((sent_index, i, j),(sent_index, i, j))
    """

    def get_head_word_id(sent_index, head_index):
        return doc[sent_index][head_index]

    features = []
    mention_indices = []

    for i, doc in enumerate(id_corpus):
        doc_features = []
        doc_indices = []

        c_mentions = []
        for s_i, sent in enumerate(cand_mentions[i]):
            for span in sent:
                c_mentions.append((s_i, span[0], span[1]))
        c_mentions.sort()

        for j, mention in enumerate(c_mentions):
            phi = []
            m_indices = []

            sent_i, bos, eos = mention
            m_head_id = get_head_word_id(sent_index=sent_i, head_index=eos)
            m_phi = emb[m_head_id]

            cands = c_mentions[:j]
            cands.reverse()
            if len(cands) > 10:
                cands = cands[:10]

            for cand in cands:
                sent_c_i, c_bos, c_eos = cand
                a_head_id = get_head_word_id(sent_index=sent_c_i, head_index=c_eos)
                a_phi = emb[a_head_id]

                phi.append(np.concatenate((m_phi, a_phi)))
                m_indices.append(((sent_i, (bos, eos)), (sent_c_i, (c_bos, c_eos))))

            doc_features.append(phi)
            doc_indices.append(m_indices)

        features.append(doc_features)
        mention_indices.append(doc_indices)

    assert len(features) == len(mention_indices)
    return features, mention_indices


def convert_into_theano_input_format(p_phi, n_phi):
    sample_x = np.concatenate((p_phi, n_phi), axis=0)
    sample_y = np.concatenate((np.ones((len(p_phi))), np.zeros((len(n_phi)))), axis=0)

    assert len(sample_x) == len(sample_y)

    return shuffle(sample_x, sample_y)
