total_phi = 0
total_phi_p = 0


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


def get_context_word_id(doc, ment, window=1):
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
    :return samples: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants; elem=phi
    :return posits: 1D: n_doc, 2D: n_ments, 3D: n_cand_ants; elem=position
    """

    samples = []
    posits = []
    for doc_ments in cand_mentions:
        sample, posit = get_mention_phi(doc_ments, test, n_cands)
        samples.append(sample)
        posits.append(posit)

    return samples, posits


def get_dist(sent_m_i, sent_a_i):
    dist = sent_m_i - sent_a_i
    assert dist > -1

    if dist < 3:
        return dist
    elif 3 <= dist < 10:
        return 3
    else:
        return 4


def get_ment_dist(ment_i, ant_i):
    dist = ment_i - ant_i
    assert dist > 0

    if dist <= 3:
        return dist + 4
    elif 3 < dist < 10:
        return 4 + 4
    else:
        return 5 + 4


def get_string_match(m_all_words, a_all_words):
    for m, a in zip(m_all_words, a_all_words):
        if m != a:
            return 6
    return 5


def get_mention_phi(cand_mentions, test=False, n_cands=10):
    samples = []
    posit = []

    """ Convert sent level into document level, and remove [] """
    c_mentions = []
    for c_ments in cand_mentions:
        for ment in c_ments:
            c_mentions.append(ment)

    for j, mention in enumerate(c_mentions):
        sample_j = []
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
        m_span_len = mention.span_len - 1

        """ Extract the candidate antecedents """
        cand_antecedents = c_mentions[:j]

        """ Extract features of the candidate antecedents """
        for k, cand_ant in enumerate(cand_antecedents):
            sent_a_i = cand_ant.sent_index
            span_a = cand_ant.span
            coref_id_a = cand_ant.coref_id

            a_all_w = cand_ant.span_words
            a_first_w = cand_ant.first_word
            a_last_w = cand_ant.last_word
            a_ctx = cand_ant.ctx
            a_span_len = cand_ant.span_len - 1

            """ Check whether gold label or not """
            label = 1 if coref_id_m == coref_id_a > -1 else 0

            """ Add samples """
            if len(sample_j) > n_cands and label == 0 and test is False:
                continue

            sample = (m_all_w + a_all_w, [m_first_w, m_last_w, a_first_w, a_last_w], m_ctx + a_ctx,
                      [get_dist(sent_m_i, sent_a_i), get_ment_dist(j, k)],
                      [m_span_len, a_span_len, get_string_match(m_all_w, a_all_w)], label)
            sample_j.append(sample)
            posit_j.append((sent_m_i, span_m, sent_a_i, span_a))

        if sample_j:
            samples.append(sample_j)
            posit.append(posit_j)

    assert len(samples) == len(posit)
    return samples, posit

