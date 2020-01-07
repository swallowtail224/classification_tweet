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
import pandas as pd


#word2vecモデルの作成と保存
def make_model(text_data, model_name):
    data = word2vec.LineSentence(text_data)
    model = word2vec.Word2Vec(data, size = 50, window=10, hs= 1, sg=1)
    model.save(model_name)
    return model


#データの読み込み
C_use_data = pd.read_csv(filepath_or_buffer="Datas/pickup_data.csv", encoding="utf_8", sep=",")
print(len(C_use_data))
C_use_data.info()
D_use_data = pd.read_csv(filepath_or_buffer="Datas/A_pickup_data.csv", encoding="utf_8", sep=",")
print(len(D_use_data))
D_use_data.info()

#テキストデータの作成
C_use_data.drop(['screen_name','user_id','tweet_id', 'tweet','postdate', 'cos_day', 'sin_day', 'tag', 'image_url', 'image', 'retweet'], axis=1, inplace=True)
C_use_data.to_csv("Datas/Word2Vec_model/dC_tweet.txt",header=False, index=False, sep=",")
D_use_data.drop(['screen_name','user_id','tweet_id', 'tweet','postdate', 'cos_day', 'sin_day', 'tag', 'image_url', 'image', 'retweet'], axis=1, inplace=True)
D_use_data.to_csv("Datas/Word2Vec_model/dD_tweet.txt",header=False, index=False, sep=",")

#モデルの作成と保存
C_model = make_model("Datas/Word2Vec_model/dC_tweet.txt", "Datas/Word2Vec_model/dC.model")
D_model = make_model("Datas/Word2Vec_model/dD_tweet.txt", "Datas/Word2Vec_model/dD.model")
