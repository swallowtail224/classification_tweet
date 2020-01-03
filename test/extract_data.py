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

# +
#retweet dataの読み込み
retweet_input = pd.read_csv(filepath_or_buffer="t_retweet.csv", encoding="utf_8", sep=",")
print(len(retweet_input))

#tweet dataの読み込み
tweet_input = pd.read_csv(filepath_or_buffer="t_tweet_data.csv", encoding="utf_8", sep=",")
print(len(tweet_input))



#retweetしたデータとしなかったデータに分割
retweet = retweet_input.query('retweet == 1')
n_retweet = retweet_input.query('retweet == 0')

#retweetしたデータ
r_data = pd.merge(retweet, tweet_input, how="inner", on="tweetID")
#retweetしなかったデータ
nr_data = pd.merge(n_retweet, tweet_input, how="inner", on="tweetID")

#余分に別れたカラムの結合
r_data['tweet2'] = r_data[['tweet', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']].apply(lambda x: '{}_{}_{}'.format(x[0], x[1], x[2],x[3],x[4]), axis=1)
nr_data['tweet2'] = nr_data[['tweet', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']].apply(lambda x: '{}_{}_{}'.format(x[0], x[1], x[2],x[3],x[4]), axis=1)

#必要なカラム以外は取り除く
r_data.drop(['date_y','tweet','Unnamed: 3', 'Unnamed: 4','Unnamed: 5', 'Unnamed: 6'], axis=1, inplace=True)
nr_data.drop(['date_y','tweet','Unnamed: 3', 'Unnamed: 4','Unnamed: 5', 'Unnamed: 6'], axis=1, inplace=True)

#不要な文字の削除
r_data["tweet2"] = r_data["tweet2"].str.replace('\n', ' ')
r_data["tweet2"] = r_data["tweet2"].str.replace('_nan', ' ')
nr_data["tweet2"] =nr_data["tweet2"].str.replace('\n', ' ')
nr_data["tweet2"] = nr_data["tweet2"].str.replace('_nan', ' ')

print(r_data.head(3))
print(nr_data.head(3))
# -

#データをランダムに5000件ずつ抽出
ran_r_data = r_data.sample(n=5000)
ran_nr_data = nr_data.sample(n=5000)
#データの結合
data = ran_r_data.append(ran_nr_data)
#dateでソート
data_s = data.sort_values('date_x')
print(data_s.head(10))

data_s.to_csv("retweet_data.txt",index=False, sep=",")

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
    #text_string = re.sub(r'([^\s\w])+', '', text_string)
    #text_string = re.sub(r'[0-9]', '', text_string)
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


text = '祝 北海道新幹線が開業 年 月 日 土 北海道の皆さん 全国の皆さん 新幹線で岩手におでんせ あっ 飛行機もアルヨ'
print(text)
text = clean_text(text)
print(text)
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
