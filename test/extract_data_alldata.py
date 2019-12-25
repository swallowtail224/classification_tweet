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
import pandas as pd
import numpy as np
import urllib.request
import unicodedata
from sklearn.model_selection import train_test_split

#tweet dataの読み込み
tweet_input = pd.read_csv(filepath_or_buffer="Datas/all_data/tweet_datas.csv", encoding="utf_8", sep=",")
print(len(tweet_input))

tweet_input.info

#NaNデータのカウント
print(tweet_input.isnull().sum())
#NaNデータの中身を確認
#tweet_input[tweet_input['tweet'].isnull()]
#NaNのデータを削除
tweet_data = tweet_input.dropna(how='any')
#NaNデータのカウント
print(tweet_data.isnull().sum())
#掲載したツイート数のカウント
published_post = tweet_data['retweet'] == 1
published_post.sum()

tweet_data.head()

#不要な文字の削除
tweet_data["tweet"] = tweet_data["tweet"].str.replace('\n', ' ').str.replace('_nan', ' ').str.replace(r'[0-9]', ' ').str.replace(r'([^\s\w])+', ' ')

tweet_data.head()

#特定のデータ数のカウント
test = ((tweet_data['screen_name'] == '川源ぶどう園(花巻市)') & (tweet_data['retweet'] == 1.0)) 
print(test.sum())

tweet_data.to_csv("Datas/all_data/all_tweet.csv",index=False, sep=",")

#postdateをdatetime形式に変換
print(pd.to_datetime(tweet_data['postdate'], format = '%Y-%m-%d'))

#テキストデータのみ取り出し、ファイルに書き出す
text = data_s.drop(['tweetID', 'retweet', 'date_x'], axis=1)
text.to_csv("tweet.txt",header=False, index=False, sep=",")
#retweetのラベルのみ取り出す
tweet_labe = data_s.drop(['tweetID', 'date_x', 'tweet2'], axis=1)
tweet_labe.to_csv("label.txt", header=Falsem index=False, sep=",")

#ツイートのテキスト読み込み
test = open("tweet.txt", "r")
lines = test.readlines()
test.close()
print(len(lines))


# +
#テキストのクリーニング
#text_data_test  = lines[9924:9929]
#print(text_data_test)
def clean_text(text_string):
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

#text_data_train = [clean_text(x) for x in text_data_test]
#print(text_data_train)


# -

#slothlibのストップワードの取得
def get_stopword():
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
    return slothlib_stopwords


stopwords = get_stopword()
print(stopwords)


#MeCabによる形態素解析
def extractter(text):
    text = unicodedata.normalize("NFKC", text)
    tagger = mc.Tagger('-Ochasen')
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
            if nes[0] == u"名詞":
                if nes[6] == u"*":
                     keyword.append(wor)
                else:
                    keyword.append(nes[6])
        node = node.next
        if node is None:
            break
    return keyword


text = '【祝】北海道新幹線が開業 | 2016年3月26日(土) -    北海道の皆さん、全国の皆さん、新幹線で岩手におでんせ～♪あっ、飛行機もアルヨ‼'
text = clean_text(text)
result = extractter(text)
print(result)


#ストップワードの除去
def except_stopwords(text, stopwords):
    for i in text:
        for j in stopwords:
            if i == j:
                text.remove(j)
    return text


s_result = except_stopwords(result, stopwords)
print(s_result)

#データセットの分割
x_train, x_test, y_train, y_test = train_test_split(namelist2_x, namelist2_y, test_size=0.3)
