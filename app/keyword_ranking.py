from collections import OrderedDict
from copy import deepcopy

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

from app import translator, nlp
from app.storage_engine import Storage


class KeywordRanking:
    """
        Extract keywords from text
        Example:
            with KeywordRanking(path='data.json', date='13-10-2019', index=0) as kr:
                kr.write_keywords()
    """

    def __init__(self, path: str = None,
                 date: str = None,
                 index: int = None,
                 text: str = None,
                 candidate_pos: list = None,
                 window_size: int = 4,
                 damp_coef: float = 0.85,
                 min_diff: float = 1e-5,
                 steps: int = 10,
                 stopwords: set = None,
                 to_lower: bool = True):

        if text and path:
            raise BaseException('text OR path should be passed, not both.')
        elif not text and not path:
            raise BaseException('path or text should be passed.')
        elif (not date or len(date.split('-')) != 3) and path:
            raise BaseException('If path exists, date should be passed in format dd-MM-yyyy.')
        elif index is None and path:
            raise BaseException('If path exists, index should be passed.')

        if text:
            text = translator.translate(text, src='ru', dest='en').text
        elif path:
            with Storage(path=path) as storage:
                text = translator.translate(storage.data[date][str(index)], src='ru', dest='en').text

        self.doc = nlp(text)
        self.date = date
        self.index = index

        self.candidate_pos = candidate_pos if candidate_pos else ['NOUN', 'PRON']
        self.window_size = window_size
        self.damp_coef = damp_coef
        self.min_diff = min_diff
        self.steps = steps
        self.stopwords = stopwords
        self.to_lower = to_lower
        self.node_weight = None

    def __enter__(self):
        return self.analyze()

    def __exit__(self, *args, **kwargs):
        self.node_weight = None

    def _set_stopwords(self) -> 'KeywordRanking':
        stop_words = STOP_WORDS.union(self.stopwords) if self.stopwords else STOP_WORDS

        for word in stop_words:
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

        return self

    def _sentence_segment(self) -> list:
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in self.doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in self.candidate_pos and token.is_stop is False:
                    if self.to_lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def _get_vocab(self, sentences: list) -> OrderedDict:
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def _get_token_pairs(self, sentences: list) -> list:
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + self.window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def _symmetrize(self, a: np.ndarray) -> np.ndarray:
        return a + a.T - np.diag(a.diagonal())

    def _get_matrix(self, vocab: OrderedDict, token_pairs: list) -> np.ndarray:
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get symmetric matrix
        g = self._symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def write_keywords(self, number: int = 10, limit: int = 1.3) -> 'KeywordRanking':
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))

        if self.date and self.index is not None:
            with Storage('statistics.json') as storage:
                data = deepcopy(storage.data)

                for i, (key, value) in enumerate(node_weight.items()):
                    if value >= limit:
                        if i > number:
                            break
                        if key not in data:
                            data[key] = {
                                'ru': translator.translate(key, src='en', dest='ru').text,
                                'date_index': [[self.date, str(self.index)]]
                            }
                        elif [self.date, str(self.index)] not in data[key]['date_index']:
                            data[key]['date_index'].append([self.date, str(self.index)])
                if data != storage.data:
                    storage.write(data=data)
        else:
            keywords = []
            for i, (key, value) in enumerate(node_weight.items()):
                if i > number:
                    break
                elif value >= limit:
                    keywords.append([key, translator.translate(key, src='en', dest='ru').text, value])
            print(f'{keywords[0]} = {keywords[1]} - {keywords[2]}')
        return self

    def analyze(self) -> 'KeywordRanking':
        """Main function to analyze text"""

        self._set_stopwords()

        # Filter sentences
        sentences = self._sentence_segment()
        vocab = self._get_vocab(sentences)
        token_pairs = self._get_token_pairs(sentences)

        # Get normalized matrix
        g = self._get_matrix(vocab, token_pairs)

        # Initialization for weight
        pr = np.array([1] * len(vocab))

        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.damp_coef) + self.damp_coef * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight

        return self
