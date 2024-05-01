import re
from typing import List


class TextProcessor:
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.regex = re.compile('[^a-zA-Z]')
        self.sym_to_ind = {alphabet[i]:i for i in range(len(alphabet))}
        self.ind_to_sym = {i:alphabet[i] for i in range(len(alphabet))}
    
    def encode(self, text: str):
        text = text.lower()
        text = text.strip()
        text = self.regex.sub(' ', text)
        return [self.sym_to_ind[sym] for sym in text]
    
    def decode(self, inds: List[int]):
        return "".join([self.ind_to_sym[ind] for ind in inds])
