import re
import random

from io_utils import UNK
from mention import Mention

import numpy as np
import theano

RE_PH = re.compile(ur'[A-Z]+')


def get_gold_mentions(corpus, limit=5, check=False):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref)
    :return: gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=Mention
    """

    gold_ments = []
    count = 0.
    max_span_len = -1
    total_span_len = 0.

    for doc_i, doc in enumerate(corpus):
        doc_ments = []

        for sent_i, sent in enumerate(doc):
            tmp_ments = get_gold_ments(doc_i, sent_i, sent)

            limited_ments = []
            for ment in tmp_ments:
                if ment.span_len <= limit:
                    limited_ments.append(ment)
                    max_span_len = ment.span_len if ment.span_len > max_span_len else max_span_len
                    total_span_len += ment.span_len

            doc_ments.append(limited_ments)
            count += len(limited_ments)

        gold_ments.append(doc_ments)

    print 'Gold Mentions: %d  Max Span Length: %d  Avg. Span Length: %f' % (count, max_span_len, total_span_len / count)

    if check:
        with open('gold_mentions.txt', 'w') as f:
            for doc, doc_ments in zip(corpus, gold_ments):
                for sent, sent_ments in zip(doc, doc_ments):
                    for ment in sent_ments:
                        print >> f, '%s %d' % (str(ment.span), ment.coref_id)
                    print >> f

                    for i, w in enumerate(sent):
                        print >> f, '%d\t%s\t%s' % (i, w[2].encode('utf-8'), w[-1].encode('utf-8'))
                    print >> f

    return gold_ments


def get_gold_ments(doc_i, sent_i, sent):
    """
    :param sent: 1D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref)
    :return: ments: 1D: n_mentions: elem=Mention
    """

    ments = []
    prev = []

    for i, w in enumerate(sent):
        mentions = w[6].split('|')

        for mention in mentions:
            if mention.startswith('('):
                if mention.endswith(')'):
                    span = (i, i)
                    coref_id = int(mention[1:-1])
                    ments.append(Mention(doc_i, sent_i, span, coref_id))
                else:
                    coref_id = int(mention[1:])
                    prev.append(((i, i), coref_id))
            else:
                if mention.endswith(')'):
                    coref_id = int(mention[:-1])

                    for j, p in enumerate(prev):
                        if coref_id == p[1]:
                            span = (p[0][0], i)
                            ments.append(Mention(doc_i, sent_i, span, coref_id))
                            prev.pop(j)
                            break
                    else:
                        print 'Error at extract_mentions(): %s' % str(sent)
                        exit()

    assert len(prev) == 0
    return ments


def get_cand_mentions(corpus, limit=5, check=False):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref_id)
    :return: cand: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=Mention
    """
    cand_ments = []
    count = 0.
    max_span_len = -1
    total_span_len = 0.

    for doc_i, doc in enumerate(corpus):
        doc_ments = []

        for sent_i, sent in enumerate(doc):
            mention_spans = []

            """ Extracting NP, Pro-Nom, NE mentions """
            mention_spans.extend(get_np(sent))
            mention_spans.extend(get_pronominals(sent))
            mention_spans.extend(get_ne(sent))

            """ Removing duplicates, and sorting """
            mention_spans = list(set(mention_spans))
            mention_spans.sort()

            tmp_ments = []
            for span in mention_spans:
                span_len = span[1] - span[0] + 1

                if span_len <= limit:
                    tmp_ments.append(Mention(doc_i, sent_i, span))

                    if span_len > max_span_len:
                        max_span_len = span_len
                    total_span_len += span_len

            doc_ments.append(tmp_ments)
            count += len(tmp_ments)

        cand_ments.append(doc_ments)

    print 'Cand Mentions: %d  Max Span Length: %d  Avg. Span Length: %f' % (count, max_span_len, total_span_len / count)

    if check:
        with open('cand_mentions.txt', 'w') as f:
            for doc, doc_ments in zip(corpus, cand_ments):
                for sent, sent_ments in zip(doc, doc_ments):
                    for ment in sent_ments:
                        print >> f, '%s' % str(ment.span)
                    print >> f

                    for sent_i, w in enumerate(sent):
                        print >> f, '%d\t%s\t%s' % (sent_i, w[2].encode('utf-8'), w[-1].encode('utf-8'))
                    print >> f

    return cand_ments


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
#    return get_max_np(spans)


def get_max_np(spans):
    max_np = []

    for symbol, bos, eos in spans:
        if symbol != 'NP':
            continue
        for t_symbol, t_bos, t_eos in spans:
            if t_symbol == symbol == 'NP':
                if t_bos < bos and eos < t_eos:
                    break
                elif t_bos == bos and eos < t_eos:
                    break
                elif t_bos < bos and eos == t_eos:
                    break
        else:
            max_np.append((bos, eos))

    return max_np


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
    :param gold: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=Mention
    :param cand: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=Mention
    """

    assert len(gold) == len(cand)

    t_count = 0
    g_total = 0
    c_total = 0

    for g_doc_ments, c_doc_ments in zip(gold, cand):

        assert len(g_doc_ments) == len(c_doc_ments)

        for g_sent_ments, c_sent_ments in zip(g_doc_ments, c_doc_ments):
            for g_ment in g_sent_ments:
                for c_ment in c_sent_ments:
                    if g_ment.span == c_ment.span:
                        t_count += 1
                        break

            g_total += len(g_sent_ments)
            c_total += len(c_sent_ments)

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


def set_cand_ment_coref(gold_ments, cand_ments):
    for g_doc_ments, c_doc_ments in zip(gold_ments, cand_ments):
        for g_sent_ments, c_sent_ments in zip(g_doc_ments, c_doc_ments):
            for c_ment in c_sent_ments:
                for g_ment in g_sent_ments:
                    if c_ment.span == g_ment.span:
                        c_ment.coref_id = g_ment.coref_id
                        break
    return cand_ments


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
    sample_y = []

    indices = []

    for sample_ments in zip(*samples):
        d_indices = []
        for sample in zip(*sample_ments):
            bos = len(sample_y)
            for s, w, c, d, y, p in zip(*sample):
                sample_s.append(s)
                sample_w.append(w)
                sample_c.append(c)
                sample_d.append(d)
                sample_y.append(y)
            d_indices.append((bos, len(sample_y)))
        indices.append(d_indices)

    assert len(sample_s) == len(sample_w) == len(sample_c) == len(sample_d) == len(sample_y)
    return [shared(sample_s), shared(sample_w), shared(sample_c), shared(sample_d), shared(sample_y)], indices
