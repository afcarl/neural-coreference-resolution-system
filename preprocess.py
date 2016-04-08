import re

from io_utils import UNK

import numpy as np
import theano

RE_PH = re.compile(ur'[A-Z]+')


def get_gold_mentions(corpus, limit=5, check=False):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref)
    :return: gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=(bos, eos)
    :return: gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions: elem=coref_id
    """

    gold_mentions = []
    gold_corefs = []
    count = 0.
    max_span_len = -1
    total_span_len = 0.

    for doc in corpus:
        doc_mention_spans = []
        doc_corefs = []

        for sent in doc:
            tmp_mention_spans, coref_ids = get_mention_spans(sent)

            mention_spans = []
            for span in tmp_mention_spans:
                span_len = span[1] - span[0] + 1

                if span_len <= limit:
                    mention_spans.append(span)

                    if span_len > max_span_len:
                        max_span_len = span_len
                    total_span_len += span_len

            doc_mention_spans.append(mention_spans)
            doc_corefs.append(coref_ids)
            count += len(mention_spans)

        assert len(doc_mention_spans) == len(doc_corefs)
        gold_mentions.append(doc_mention_spans)
        gold_corefs.append(doc_corefs)

    assert len(gold_mentions) == len(gold_corefs)
    print 'Gold Mentions: %d  Max Span Length: %d  Avg. Span Length: %f' % (count, max_span_len, total_span_len / count)

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


def get_cand_mentions(corpus, limit=5, check=False):
    """
    :param corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=(doc_id, part_id, word, tag, syn, ne, coref_id)
    :return: cand: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    """
    cand_mentions = []
    count = 0.
    max_span_len = -1
    total_span_len = 0.

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

            new = []
            for span in mention:
                span_len = span[1] - span[0] + 1

                if span_len <= limit:
                    new.append(span)

                    if span_len > max_span_len:
                        max_span_len = span_len
                    total_span_len += span_len

            mentions.append(new)
            count += len(new)

        cand_mentions.append(mentions)

    print 'Cand Mentions: %d  Max Span Length: %d  Avg. Span Length: %f' % (count, max_span_len, total_span_len / count)

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
    :param id_corpus: 1D: n_doc, 2D; n_sents, 3D: n_words; elem=word_id
    :param cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    :param gold_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=(bos, eos)
    :param gold_corefs: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=coref_id
    :return: x_span: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants, 4D: limit * 2; elem=word id
    :return: x_word: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants, 4D: (m_first, m_last, a_first, a_last); elem=word id
    :return: x_ctx: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants, 4D: window * 2 * 2; elem=word id
    :return: x_dist: 1D: n_doc, 2D: n_mentions, 3D: n_cand_ants; elem=sent dist
    :return: y: 1D: n_doc, 2D: n_mentions; elem=0/1
    """

    x_span = []
    x_word = []
    x_ctx = []
    x_dist = []
    y = []
    posit = []

    for i, doc in enumerate(id_corpus):
        # x_i: 1D: n_phi, 2D: window * 2
        # y_i: 1D: n_phi; elem=oracle flag (1=true, 0=false)

        if test is False:
            x_span_i, x_word_i, x_ctx_i, x_dist_i, y_i, posit_i =\
                get_gold_mention_features(doc=doc, gold_mentions=gold_mentions[i], gold_corefs=gold_corefs[i], window=window)

            x_span.append(x_span_i)
            x_word.append(x_word_i)
            x_ctx.append(x_ctx_i)
            x_dist.append(x_dist_i)
            y.append(y_i)
            posit.append(posit_i)

        x_span_i, x_word_i, x_ctx_i, x_dist_i, y_i, posit_i =\
            get_cand_mention_features(doc=doc, cand_mentions=cand_mentions[i], gold_mentions=gold_mentions[i],
                                      gold_corefs=gold_corefs[i], test=test, window=window)

        x_span.append(x_span_i)
        x_word.append(x_word_i)
        x_ctx.append(x_ctx_i)
        x_dist.append(x_dist_i)
        y.append(y_i)
        posit.append(posit_i)

    return x_span, x_word, x_ctx, x_dist, y, posit


def get_context_word_id(doc, sent_index, span, window=5):
    bos = span[0]
    eos = span[1]

    """ prev context """
    sent = doc[sent_index][:bos]
    while len(sent) < window and sent_index > 0:
        sent_index -= 1
        sent = doc[sent_index] + sent
    if len(sent) < window:
        sent = [0 for i in xrange(window-len(sent))] + sent
    prev_ctx = sent[-window:]

    """ post context """
    doc_len = len(doc)
    sent = doc[sent_index][eos+1:]
    while len(sent) < window and sent_index < doc_len-1:
        sent_index += 1
        sent = sent + doc[sent_index]
    if len(sent) < window:
        sent = sent + [0 for i in xrange(window-len(sent))]
    post_ctx = sent[: window]

    assert len(prev_ctx) == len(post_ctx) == window

    return prev_ctx + post_ctx


def get_mention_word_id(doc, sent_index, span, limit=5):
    bos = span[0]
    eos = span[1]
    span_len = eos - bos + 1
    sent = doc[sent_index]
    padded_span = sent[bos: eos+1] + [0 for i in xrange(limit - span_len)]
    first_word = sent[bos]
    last_word = sent[eos]
    assert len(padded_span) == limit
    return padded_span, first_word, last_word


def get_dist(sent_m_i, sent_a_i):
    dist = sent_m_i - sent_a_i
    assert dist > -1
    return dist if dist < 10 else 10


def get_gold_mention_features(doc, gold_mentions, gold_corefs, window=5):
    """
    :param doc: 1D; n_sents, 2D: n_words; elem=word_id
    :param gold_mentions: 1D: n_sents, 2D: n_mentions; elem=(bos, eos)
    :param gold_corefs: 1D: n_sents, 2D: n_mentions; elem=coref_id
    :return: x_word: 1D: n_mentions, 2D: n_window * 2 (ment & ant context)
    :return: x_dist: 1D: n_mentions; 2D: n_gold_ants; elem=distance
    :return: y: 1D: n_mentions; 2D: n_gold_ants; elem=label (1)
    """

    def get_antecedents(coref_id, antecedents):
        antecedents.reverse()
        return [ant for ant in antecedents if ant[-1] == coref_id]

    x_span = []  # input: all word ids of spans
    x_word = []  # input: first and last word ids of spans
    x_ctx = []  # input: word ids of context surrounding spans
    x_dist = []  # input: distances between the two sentences containing ment & ant
    y = []  # output: labels; 1 if the sample is gold else 0
    positions = []  # ment & ant positions within a document

    """ Convert sent level into document level, and remove [] """
    g_mentions = []
    for s_i, sent in enumerate(zip(gold_mentions, gold_corefs)):
        if sent[0]:
            for span, coref in zip(*sent):
                g_mentions.append((s_i, span, coref))
    g_mentions.sort()

    """ Extract features """
    for j, mention in enumerate(g_mentions):
        x_span_j = []
        x_word_j = []
        x_ctx_j = []
        x_dist_j = []
        position_j = []

        sent_m_i, span_m, _ = mention

        """ Extract features of the target mention """
        # all_w: 1D: limit, first_w: int, last_m: int
        m_all_w, m_first_w, m_last_w = get_mention_word_id(doc=doc, sent_index=sent_m_i, span=span_m)
        m_ctx = get_context_word_id(doc=doc, sent_index=sent_m_i, span=span_m, window=window)  # 1D: window * 2

        """ Extract features of the gold antecedents """
        gold_antecedents = get_antecedents(coref_id=mention[-1], antecedents=g_mentions[:j])
        for gold_ant in gold_antecedents:
            sent_a_i, span_a, _ = gold_ant
            a_all_w, a_first_w, a_last_w = get_mention_word_id(doc=doc, sent_index=sent_a_i, span=span_a)
            a_ctx = get_context_word_id(doc=doc, sent_index=sent_a_i, span=span_a, window=window)

            x_span_j.append(m_all_w + a_all_w)
            x_word_j.append([m_first_w, m_last_w, a_first_w, a_last_w])
            x_ctx_j.append(m_ctx + a_ctx)
            x_dist_j.append(get_dist(sent_m_i, sent_a_i))
            position_j.append((sent_m_i, span_m, sent_a_i, span_a))

        if x_ctx_j:
            x_span.append(x_span_j)
            x_word.append(x_word_j)
            x_ctx.append(x_ctx_j)
            x_dist.append(x_dist_j)
            y.append([1 for i in xrange(len(x_ctx_j))])
            positions.append(position_j)

    assert len(x_span) == len(x_word) == len(x_ctx) == len(x_dist) == len(y) == len(positions)
    return x_span, x_word, x_ctx, x_dist, y, positions


def get_cand_mention_features(doc, cand_mentions, gold_mentions, gold_corefs, test=False, window=5):
    x_span = []
    x_word = []
    x_ctx = []
    x_dist = []
    y = []
    posit = []

    """ Convert sent level into document level, and remove [] """
    c_mentions = []
    for s_i, sent in enumerate(cand_mentions):
        for span_m in sent:
            c_mentions.append((s_i, span_m[0], span_m[1]))
    c_mentions.sort()

    for j, mention in enumerate(c_mentions):
        x_span_j = []
        x_word_j = []
        x_ctx_j = []
        x_dist_j = []
        y_j = []
        posit_j = []

        sent_m_i, bos, eos = mention
        span_m = (bos, eos)

        g_mention_spans = gold_mentions[sent_m_i]
        g_coref_ids = gold_corefs[sent_m_i]

        m_gold_flag = False
        m_coref_id = -1

        if span_m in g_mention_spans:
            m_gold_flag = True
            m_coref_id = g_coref_ids[g_mention_spans.index(span_m)]

        """ Extract features of the target mention """
        m_all_w, m_first_w, m_last_w = get_mention_word_id(doc=doc, sent_index=sent_m_i, span=span_m)
        m_ctx = get_context_word_id(doc=doc, sent_index=sent_m_i, span=span_m, window=window)  # 1D: window * 2

        """ Extract the candidate antecedents """
        cand_antecedents = c_mentions[:j]
        cand_antecedents.reverse()

        """ Filter the number of the candidate antecedents """
        if len(cand_antecedents) > 2 and test is False:
            cand_antecedents = cand_antecedents[:2]

        """ Extract features of the candidate antecedents """
        for cand_ant in cand_antecedents:
            sent_a_i, a_bos, a_eos = cand_ant
            span_a = (a_bos, a_eos)

            """ Check errors """
            if sent_a_i < sent_m_i:
                pass
            elif sent_a_i == sent_m_i and a_bos > bos:
                print 'Error: sent:%d span(%d,%d), sent:%d span(%d,%d)' % (sent_m_i, bos, eos, sent_a_i, a_bos, a_eos)
                exit()
            elif sent_a_i > sent_m_i:
                print 'Error: sent:%d span(%d,%d), sent:%d span(%d,%d)' % (sent_m_i, bos, eos, sent_a_i, a_bos, a_eos)
                exit()

            a_all_w, a_first_w, a_last_w = get_mention_word_id(doc=doc, sent_index=sent_a_i, span=span_a)
            a_ctx = get_context_word_id(doc=doc, sent_index=sent_a_i, span=span_a, window=window)

            g_mention_spans = gold_mentions[sent_a_i]
            g_coref_ids = gold_corefs[sent_a_i]

            a_gold_flag = False
            a_coref_id = -1

            """ Check whether gold span or not """
            if span_a in g_mention_spans:
                a_gold_flag = True
                a_coref_id = g_coref_ids[g_mention_spans.index(span_a)]

            if m_gold_flag and a_gold_flag and m_coref_id == a_coref_id:
                if test:
                    x_span_j.append(m_all_w + a_all_w)
                    x_word_j.append([m_first_w, m_last_w, a_first_w, a_last_w])
                    x_ctx_j.append(m_ctx + a_ctx)
                    x_dist_j.append(get_dist(sent_m_i, sent_a_i))
                    y_j.append(1)
                    posit_j.append((sent_m_i, span_m, sent_a_i, span_a))
            else:
                x_span_j.append(m_all_w + a_all_w)
                x_word_j.append([m_first_w, m_last_w, a_first_w, a_last_w])
                x_ctx_j.append(m_ctx + a_ctx)
                x_dist_j.append(get_dist(sent_m_i, sent_a_i))
                y_j.append(0)
                posit_j.append((sent_m_i, span_m, sent_a_i, span_a))

        if x_ctx_j:
            x_span.append(x_span_j)
            x_word.append(x_word_j)
            x_ctx.append(x_ctx_j)
            x_dist.append(x_dist_j)
            y.append(y_j)
            posit.append(posit_j)

    assert len(x_span) == len(x_word) == len(x_ctx) == len(x_dist) == len(y) == len(posit)
    return x_span, x_word, x_ctx, x_dist, y, posit


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
        for sample in zip(*sample_ments):
            bos = len(sample_y)
            for s, w, c, d, y, p in zip(*sample):
                sample_s.append(s)
                sample_w.append(w)
                sample_c.append(c)
                sample_d.append(d)
                sample_y.append(y)
            indices.append((bos, len(sample_y)))

    assert len(sample_s) == len(sample_w) == len(sample_c) == len(sample_d) == len(sample_y)
    return [shared(sample_s), shared(sample_w), shared(sample_c), shared(sample_d), shared(sample_y)], indices


"""
def theano_format(x_span, x_word, x_ctx, x_dist, y):
    def t_format(sample):
        return [[np.asarray(j, dtype='int32') for j in i] for i in sample]

    return t_format(x_span), t_format(x_word), t_format(x_ctx), t_format(x_dist), t_format(y)
"""