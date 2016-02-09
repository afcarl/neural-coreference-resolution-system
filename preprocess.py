__author__ = 'hiroki'

import re

from io import UNK
from utils import shuffle

import numpy as np

RE_PH = re.compile(ur'[A-Z]+')
NON_ANA = '-'


def get_gold_mentions(corpus, check=False):
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

    if check:
        with open('gold_mentions.txt', 'w') as f:
            for doc, doc_mentions, doc_corefs in zip(corpus, gold_mentions, gold_corefs):
                for sent, sent_mentions, sent_corefs in zip(doc, doc_mentions, doc_corefs):
                    print >> f, '%s %s' % (str(sent_mentions), str(sent_corefs))
                    for i, w in enumerate(sent):
                        print >> f, '%d\t%s\t%s' % (i, w[2].encode('utf-8'), w[-1].encode('utf-8'))
                    print >> f

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


def get_cand_mentions(corpus, check=False):
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
            mention.sort()
            mentions.append(mention)
            count += len(mention)

        cand_mentions.append(mentions)

    print 'Cand Mentions: %d' % count

    if check:
        with open('cand_mentions.txt', 'w') as f:
            for doc, doc_mentions in zip(corpus, cand_mentions):
                for sent, sent_mentions in zip(doc, doc_mentions):
                    print >> f, '%s' % str(sent_mentions)
                    for i, w in enumerate(sent):
                        print >> f, '%d\t%s\t%s' % (i, w[2].encode('utf-8'), w[-1].encode('utf-8'))
                    print >> f

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


def get_features(id_corpus, cand_mentions, gold_mentions=None, gold_corefs=None, test=False, window=5):
    """
    :param window:
    :param gold_corefs:
    :param id_corpus: 1D: n_doc, 2D; n_sents, 3D: n_words; elem=word_id
    :param gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    :param cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    :return: features: 1D: n_doc, 2D: n_sents, 3D: n_words * 2
    """

#    if gold_mentions:
#        test = False
#    else:
#        test = True

    x = []
    y = []
    posit = []

    for i, doc in enumerate(id_corpus):
        # x_i: 1D: n_phi, 2D: window * 2
        # y_i: 1D: n_phi; elem=oracle flag (1=true, 0=false)

        if test is False:
            x_i, y_i, posit_i = get_gold_mention_features(doc=doc, gold_mentions=gold_mentions[i],
                                                          gold_corefs=gold_corefs[i], window=window)
            x.append(x_i)
            y.append(y_i)
            posit.append(posit_i)

        x_i, y_i, posit_i = get_cand_mention_features(doc=doc, cand_mentions=cand_mentions[i],
                                                      gold_mentions=gold_mentions[i], gold_corefs=gold_corefs[i],
                                                      test=test, window=window)
        x.append(x_i)
        y.append(y_i)
        posit.append(posit_i)

    return x, y, posit


def get_context_word_id(doc, sent_index, head_index, slide, pad):
    head_index += slide
    padded_sent = pad + doc[sent_index] + pad
    return padded_sent[head_index - slide: head_index + slide + 1]


def get_gold_mention_features(doc, gold_mentions, gold_corefs, window=5):
    def get_antecedents(coref_id, antecedents):
        antecedents.reverse()
        return [ant for ant in antecedents if ant[-1] == coref_id]

    slide = window / 2
    pad = [0 for k in xrange(slide)]

    x = []
    y = []
    posit = []

    """ Convert sent level into document level, and remove [] """
    g_mentions = []
    for s_i, sent in enumerate(zip(gold_mentions, gold_corefs)):
        if sent[0]:
            for span, coref in zip(*sent):
                g_mentions.append((s_i, span, coref))
    g_mentions.sort()

    """ Extract features """
    for j, mention in enumerate(g_mentions):
        x_j = []
        posit_j = []

        sent_i, span, _ = mention

        """ Extract features of the target mention """
        mention_context = get_context_word_id(doc=doc, sent_index=sent_i, head_index=span[1], slide=slide, pad=pad)

        """ Extract features of the gold antecedents """
        gold_antecedents = get_antecedents(coref_id=mention[-1], antecedents=g_mentions[:j])
        for gold_ant in gold_antecedents:
            sent_a_i, span_a, _ = gold_ant
            ant_context = get_context_word_id(doc=doc, sent_index=sent_a_i, head_index=span_a[1], slide=slide, pad=pad)

            x_j.append(mention_context + ant_context)
            posit_j.append((sent_i, span, sent_a_i, span_a))

        if x_j:
            x.append(x_j)
            y.append([1 for i in xrange(len(x_j))])
            posit.append(posit_j)

    assert len(x) == len(y) == len(posit)
    return x, y, posit


def get_cand_mention_features(doc, cand_mentions, gold_mentions, gold_corefs, test=False, window=5):
    slide = window / 2
    pad = [0 for k in xrange(slide)]

    x = []
    y = []
    posit = []

    """ Convert sent level into document level, and remove [] """
    c_mentions = []
    for s_i, sent in enumerate(cand_mentions):
        for span in sent:
            c_mentions.append((s_i, span[0], span[1]))
    c_mentions.sort()

    for j, mention in enumerate(c_mentions):
        x_j = []
        y_j = []
        posit_j = []

        sent_i, bos, eos = mention
        span = (bos, eos)

        g_mention_spans = gold_mentions[sent_i]
        g_coref_ids = gold_corefs[sent_i]

        m_gold_flag = False
        m_coref_id = -1

        if span in g_mention_spans:
            m_gold_flag = True
            m_coref_id = g_coref_ids[g_mention_spans.index(span)]

        """ Extract features of the target mention """
        mention_context = get_context_word_id(doc=doc, sent_index=sent_i, head_index=bos, slide=slide, pad=pad)

        """ Extract the candidate antecedents """
        cand_antecedents = c_mentions[:j]
        cand_antecedents.reverse()

        """ Filter the number of the candidate antecedents """
        if len(cand_antecedents) > 2:
            cand_antecedents = cand_antecedents[:2]

        """ Extract features of the candidate antecedents """
        for cand_ant in cand_antecedents:
            sent_c_i, c_bos, c_eos = cand_ant
            ant_context = get_context_word_id(doc=doc, sent_index=sent_c_i, head_index=c_eos, slide=slide, pad=pad)

            span_c = (c_bos, c_eos)

            g_mention_spans = gold_mentions[sent_c_i]
            g_coref_ids = gold_corefs[sent_c_i]

            c_gold_flag = False
            c_coref_id = -1

            if span_c in g_mention_spans:
                c_gold_flag = True
                c_coref_id = g_coref_ids[g_mention_spans.index(span_c)]

            if m_gold_flag and c_gold_flag and m_coref_id == c_coref_id:
                if test:
                    x_j.append(mention_context + ant_context)
                    y_j.append(1)
                    posit_j.append((sent_i, span, sent_c_i, span_c))
            else:
                x_j.append(mention_context + ant_context)
                y_j.append(0)
                posit_j.append((sent_i, span, sent_c_i, span_c))

        if x_j:
            x.append(x_j)
            y.append(y_j)
            posit.append(posit_j)

    assert len(x) == len(y) == len(posit)
    return x, y, posit


def convert_into_theano_input_format(x, y):
    sample_x = [[np.asarray(x_j, dtype='int32') for x_j in x_i] for x_i in x]
    sample_y = [[np.asarray(y_j, dtype='int32') for y_j in y_i] for y_i in y]
    return sample_x, sample_y
