from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import data
import sys

path = None
review_len = data.review_len
# Здесь необходимо в переменную path получить
# путь до файла, который был передан при запуске
path = sys.argv[1]

with open(path) as f:
    file_data = f.read()

arr = file_data
batch = data.encode_batch(arr)
model = load_model('saved_model_cnn_lstm_drop.h5')
result = model.predict(batch, verbose=0)

print("----------------------------------------------------\n")

print("Вероятность того, что обзор положительный: %.2f%%" % (result*100))
quit()

