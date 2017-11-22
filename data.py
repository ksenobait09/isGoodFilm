import pandas as pd
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
import numpy as np

word_dict = imdb.get_word_index()
inv_map = {v: k for k, v in word_dict.items()}

top_words = 5000
review_len = 500

def encode_review(text):
    result = []
    arr = text_to_word_sequence(text, lower=True, split=" ")
    for word in arr:
        w = encode_word(word)
        if w is not None and w <= top_words:
            result.append(w)
    return result

def encode_word(word):
    if word not in word_dict:
        return 0
    return word_dict[word]

def decode_word(ind):
    if ind not in inv_map:
        return None
    return inv_map[ind]
    
def encode_batch(arr):
    arr = encode_review(arr)
    return sequence.pad_sequences([arr], maxlen=review_len)
