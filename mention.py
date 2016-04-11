class Mention(object):
    def __init__(self, doc_index, sent_index, span, coref_id=-1):
        self.doc_index = doc_index
        self.sent_index = sent_index
        self.span = span
        self.coref_id = coref_id
        self.span_len = self.span[1] - self.span[0] + 1

        self.span_words = []
        self.first_word = -1
        self.last_word = -1
        self.ctx = []


