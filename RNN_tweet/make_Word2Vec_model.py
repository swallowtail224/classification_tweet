# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from gensim.models import word2vec
import numpy as np


# +
#word2vecモデルの作成と保存
def make_model(text_data, model_name):
    data = word2vec.LineSentence(text_data)
    model = word2vec.Word2Vec(data, size = 50, window=10, hs= 1, sg=1)
    model.save(model_name)
    return model

#学習したモデルから各文章のベクトル化
def document_vector(text, model, num_features):
    model = word2vec.Word2Vec.load(model)
    bag_of_centroids = np.zeros(num_features, dtype = 'float32')
    
    for word in text:
        try:
            temp = model[word]
            #print(temp)
        except:
            continue
        bag_of_centroids += temp
        #print("//")
        #print(bag_of_centroids)
        
    bag_of_centroids = bag_of_centroids / len(text)
    return bag_of_centroids


# -

N_model = make_model("Datas/N_extract_tweet.txt", "Datas/50_noun_tweet.model")
A_model = make_model("Datas/A_extract_tweet.txt", "Datas/50_other_tweet.model")

#テスト
model = word2vec.Word2Vec.load('Datas/noun_tweet.model')
index = model.wv.index2word
print(model['陸前高田'])


def read_data(file):
    f = open(file, "r", encoding="utf-8")
    datas = f.readlines()
    f.close()
    return datas


N_model = "Datas/noun_tweet.model"
wordlen = 200
N_texts = read_data("Datas/N_extract_tweet.txt")
WN_data = [document_vector(a, N_model, wordlen) for a in N_texts]
np_WN_data = np.array(WN_data)

# +
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
#ラベル読み込み
test = open("Datas/label.txt", "r", encoding="utf-8")
label = test.readlines()
test.close()
print(len(label))

training_samples = 8000 # training data 80 : validation data 20
validation_samples = 10000 - training_samples

# バイナリの行列に変換
categorical_labels = to_categorical(label)
labels = np.asarray(categorical_labels)

x_train = np_WN_data[:training_samples]
y_train = labels[:training_samples]
x_val = np_WN_data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
# -

np_WN_data = np.array(WN_data)

print(type(np_WN_data))

# +
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.core import Dropout

model = Sequential()
model.add(Embedding(20000, 100, input_length=200))
model.add(LSTM(1000))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

early_stopping = EarlyStopping(patience=0, verbose=1)
# -

history = model.fit(x_train, y_train,
                    epochs=15, 
                    batch_size=250, 
                    validation_split=0.2, 
                    validation_data=(x_val, y_val))

print(x_train[0])


