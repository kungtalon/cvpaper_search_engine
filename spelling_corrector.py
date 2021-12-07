from functools import lru_cache
import re
import pickle as pkl
from nltk.stem import PorterStemmer
from collections import defaultdict

class SpellingCorrector:
    def __init__(self, corpus_pkl):
        with open(corpus_pkl, 'rb') as f:
            corpus = pkl.load(f)
        self.vocab = defaultdict(int)
        self.stemmer = PorterStemmer()
        for sent in corpus:
            for term in sent:
                self.vocab[term] += 1
    
    def words(self, text):
        return re.findall(r'\w+', text.lower())

    @lru_cache(maxsize = 256)
    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=lambda x: self.vocab.get(x, 0))

    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [''])

    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if self.stemmer.stem(w) in self.vocab)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))