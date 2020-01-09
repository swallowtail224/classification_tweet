# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
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

#slothlibのストップワードの取得
def get_stopword():
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
    return slothlib_stopwords

#ストップワードの除去
def except_stopwords(text):
    stopwords = get_stopword()
    for i in text:
        for j in stopwords:
            if i == j:
                text.remove(j)
    return text


# -

#データの確認
base_data = pd.read_csv(filepath_or_buffer="Datas/pickup_data.csv", encoding="utf_8", sep=",")
base_data.info()
base_data.head()

# +
#user_idの固有データの取り出し
nu = base_data['user_id'].unique()
print(nu)
#データフレームからuser_idを抽出
ids = base_data['user_id']
#0~31の数値へ変換
post_user = []
for i in ids:
    for j in range(len(nu)):
        if i == nu[j]:
            post_user.append(j)

#データフレームへ変換
s = pd.DataFrame(post_user, columns=['user_id2'])
#元のデータと結合
#n_base_data = base_data.join(s)
# -

#不要な文字の削除
tag_list = base_data["tag"].str.replace(r'[0-9]', ' ').str.replace(r'[０-９]', ' ').str.replace(r'([^\s\w])+', ' ')
#形態素解析
for i in range(len(tag_list)):
    tag_list[i] = extractter(tag_list[i])
    tag_list[i] = except_stopwords(tag_list[i])

tag_list[2]

#元のデータと結合
n_base_data = base_data.join(s)
n_base_data['tag2'] = tag_list

n_base_data.head()

#データ一時保存
n_base_data.to_csv("Datas/multi_data.csv",index=False, sep=",")
#データの確認
b_data = pd.read_csv(filepath_or_buffer="Datas/multi_data.csv", encoding="utf_8", sep=",")
b_data.info()
b_data.head()

#不要文字の置き換え
b_data['tag2'] = b_data['tag2'].str.replace(r'([^\s\w])+', '')

#データ再保存
b_data.to_csv("Datas/multi_data.csv",index=False, sep=",")


