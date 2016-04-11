def set_word_id_for_ment(id_corpus, ments):
    """
    :param id_corpus: 1D: n_doc, 2D: n_sents, 3D: n_words; elem=word id
    :param ments: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=Mention
    :return: ments: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=Mention
    """

    for doc_ments in ments:
        for sent_ments in doc_ments:
            for ment in sent_ments:
                doc = id_corpus[ment.doc_index]
                sent = doc[ment.sent_index]
                span_words, f_word, l_word = get_mention_word_id(sent, ment)

                ment.span_words = span_words
                ment.first_word = f_word
                ment.last_word = l_word
                ment.ctx = get_context_word_id(doc, ment)
    return ments


def get_mention_word_id(sent, ment, limit=5):
    bos = ment.span[0]
    eos = ment.span[1]
    span_words = sent[bos: eos+1] + [0 for i in xrange(limit - ment.span_len)]
    first_word = sent[bos]
    last_word = sent[eos]
    assert len(span_words) == limit
    return span_words, first_word, last_word


def get_context_word_id(doc, ment, window=5):
    bos = ment.span[0]
    eos = ment.span[1]

    """ prev context """
    sent_index = ment.sent_index
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


def get_features(cand_mentions, test=False, n_cands=10):
    """
    :param cand_mentions: 1D: n_doc, 2D: n_sents, 3D: n_mentions; elem=Mention
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

    for doc_ments in cand_mentions:
        x_span_i, x_word_i, x_ctx_i, x_dist_i, y_i, posit_i = get_mention_phi(doc_ments, test, n_cands)

        x_span.append(x_span_i)
        x_word.append(x_word_i)
        x_ctx.append(x_ctx_i)
        x_dist.append(x_dist_i)
        y.append(y_i)
        posit.append(posit_i)

    return x_span, x_word, x_ctx, x_dist, y, posit


def get_dist(sent_m_i, sent_a_i):
    dist = sent_m_i - sent_a_i
    assert dist > -1
    return dist if dist < 10 else 10


def get_mention_phi(cand_mentions, test=False, n_cands=10):
    x_span = []
    x_word = []
    x_ctx = []
    x_dist = []
    y = []
    posit = []

    """ Convert sent level into document level, and remove [] """
    c_mentions = []
    for c_ments in cand_mentions:
        for ment in c_ments:
            c_mentions.append(ment)

    for j, mention in enumerate(c_mentions):
        x_span_j = []
        x_word_j = []
        x_ctx_j = []
        x_dist_j = []
        y_j = []
        posit_j = []

        sent_m_i = mention.sent_index
        span_m = mention.span
        coref_id_m = mention.coref_id

        """ Extract features of the target mention """
        # all_w: 1D: limit, first_w: int, last_m: int
        m_all_w = mention.span_words
        m_first_w = mention.first_word
        m_last_w = mention.last_word
        m_ctx = mention.ctx

        """ Extract the candidate antecedents """
        cand_antecedents = c_mentions[:j]

        """ Extract features of the candidate antecedents """
        for cand_ant in cand_antecedents:
            sent_a_i = cand_ant.sent_index
            span_a = cand_ant.span
            coref_id_a = cand_ant.coref_id

            a_all_w = cand_ant.span_words
            a_first_w = cand_ant.first_word
            a_last_w = cand_ant.last_word
            a_ctx = cand_ant.ctx

            """ Check whether gold label or not """
            label = 1 if coref_id_m == coref_id_a and coref_id_m > -1 else 0

            """ Add samples """
            if len(y_j) > n_cands and label == 0 and test is False:
                continue

            x_span_j.append(m_all_w + a_all_w)
            x_word_j.append([m_first_w, m_last_w, a_first_w, a_last_w])
            x_ctx_j.append(m_ctx + a_ctx)
            x_dist_j.append(get_dist(sent_m_i, sent_a_i))
            y_j.append(label)
            posit_j.append((sent_m_i, span_m, sent_a_i, span_a))

        if x_span_j:
            x_span.append(x_span_j)
            x_word.append(x_word_j)
            x_ctx.append(x_ctx_j)
            x_dist.append(x_dist_j)
            y.append(y_j)
            posit.append(posit_j)

    assert len(x_span) == len(x_word) == len(x_ctx) == len(x_dist) == len(y) == len(posit)
    return x_span, x_word, x_ctx, x_dist, y, posit

