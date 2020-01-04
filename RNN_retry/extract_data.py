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

import MeCab as mc
import re
import urllib.request
import unicodedata
import pandas as pd
import numpy as np
import math


# +
#slothlibのストップワードの取得
def get_stopword():
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
    return slothlib_stopwords

#ストップワードの追加
def add_stopword(slothlib_stopwords):
    s = open("Datas/stopwords.txt", "r", encoding="utf-8")
    stop = s.readlines()
    s.close()
    stop = [a.strip() for a in stop]
    slothlib_stopwords += stop
    return slothlib_stopwords

#ストップワードの除去
def except_stopwords(text):
    stopwords = get_stopword()
    ex_stopwords = add_stopword(stopwords)
    for i in text:
        for j in ex_stopwords:
            if i == j:
                text.remove(j)
    return text

#テキストのクリーニング
def clean_text(text_string):
    text_string = re.sub(r'[０-９]', '', text_string)
    text_string = text_string.lower()
    return(text_string)

#MeCabによる形態素解析
def extractter(text, flag):
    text = unicodedata.normalize("NFKC", text)
    tagger = mc.Tagger(r'-Ochasen')
    tagger.parse('')
    node = tagger.parseToNode(text)
    key= []
    word = []
    keyword =[]
    while node:
        nes = node.feature.split(",")
        wor = node.surface
        if wor != "":
            nes = node.feature.split(",")
            if flag == 0:
                if nes[0] == u"名詞":
                    if nes[6] == u"*":
                         keyword.append(wor)
                    else:
                        keyword.append(nes[6])
            else:
                if nes[0] == u"名詞":
                    if nes[6] == u"*":
                         keyword.append(wor)
                    else:
                        keyword.append(nes[6])
                elif nes[0] == u"動詞":
                    if nes[6] == u"*":
                        keyword.append(wor)
                    else:
                        keyword.append(nes[6])
                elif nes[0] == u"形容詞":
                    if nes[6] == u"*":
                        keyword.append(wor)
                    else:
                        keyword.append(nes[6])
                elif nes[0] == u"副詞":
                    if nes[6] == u"*":
                        keyword.append(wor)
                    else:
                        keyword.append(nes[6])
        node = node.next
        if node is None:
            break
    return keyword

#テキストの結合
def join_data(text):
    texts = ' '.join(text)
    return texts

#画像URLをあるかないかの2値に変換
def conversion_image(x):
    if x is not "0":
        return 1
    else:
        return 0

#日付データをsin, consへ変換
def change_cos(x):
    return np.cos(math.radians(90 - (x / 365)*360))

def change_sin(x):
    return np.sin(math.radians(90 - (x / 365)*360))

def change_date(df,col):
    df['cos_day']=df[col].dt.dayofyear
    df['cos_day'] =df['cos_day'].apply(change_cos)
    df['sin_day']=df[col].dt.dayofyear
    df['sin_day'] =df['sin_day'].apply(change_sin)
    return df


# -

#メイン
#データの読み込み
data_input = pd.read_csv(filepath_or_buffer="Datas/all_data/all_data.csv", encoding="utf_8", sep=",")
print(len(data_input))

data_input.head()

data_input['image'] = data_input['image_url'].apply(conversion_image)
data_input.head()

#postdateをdatetime形式に変換して、sin,cosに変換
data_input['postdate'] = pd.to_datetime(data_input['postdate'], format = '%Y-%m-%d')
data_input = change_date(data_input,col="postdate")
data_input.head()

#テキストのクリーング
data_input['tweet2'] = data_input['tweet'].apply(clean_text)
#形態素解析
x = 0
data_input['tweet2'] = data_input['tweet'].apply(lambda text:extractter(text, 0))
#ストップワードの除去
data_input['tweet2'] = data_input['tweet2'].apply(except_stopwords)
#リストの連結
data_input['tweet2'] = data_input['tweet2'].apply(join_data)
data_input.head(20)

ch_data_input = data_input.loc[:, ['screen_name', 'user_id', 'tweet_id', 'tweet', 'tweet2', 'postdate', 'cos_day', 'sin_day', 'tag', 'image_url', 'image', 'retweet']]
ch_data_input.head()

ch_data_input.to_csv("Datas/all_data/extract_allData.csv",index=False, sep=",")

#データの確認
test = pd.read_csv(filepath_or_buffer="Datas/all_data/extract_allData.csv", encoding="utf_8", sep=",")
print(len(test))

test.info()
