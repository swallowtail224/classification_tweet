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

import pandas as pd
import numpy as np


def make_data(data, text_data):
    n_data = pd.merge(data, text_data, how="inner", on="tweetID")
    #余分に別れたカラムの結合
    n_data['tweet2'] = n_data[['tweet', 'Unnamed: 3', 'Unnamed: 4', 
                               'Unnamed: 5', 'Unnamed: 6']].apply(lambda x: '{}_{}_{}'.format(x[0], x[1], x[2],x[3],x[4]), 
                                                                  axis=1)
    #必要なカラム以外は取り除く
    n_data.drop(['date_y','tweet','Unnamed: 3', 'Unnamed: 4','Unnamed: 5', 'Unnamed: 6'], axis=1, inplace=True)
    #不要な文字の削除
    n_data["tweet2"] = n_data["tweet2"].str.replace('\n', ' ')
    n_data["tweet2"] = n_data["tweet2"].str.replace('_nan', ' ')
    return n_data


# +
#retweet dataの読み込み
retweet_input = pd.read_csv(filepath_or_buffer="Datas/t_retweet.csv", encoding="utf_8", sep=",")
print(len(retweet_input))

#tweet dataの読み込み
tweet_input = pd.read_csv(filepath_or_buffer="Datas/t_tweet_data.csv", encoding="utf_8", sep=",")
print(len(tweet_input))

#retweetしたデータとしなかったデータに分割
retweet = retweet_input.query('retweet == 1')
n_retweet = retweet_input.query('retweet == 0')

r_data = make_data(retweet, tweet_input)
nr_data = make_data(n_retweet, tweet_input)

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
print(data_s.head(3))

#テキストデータのみ取り出し、ファイルに書き出す
text = data_s.drop(['tweetID', 'retweet', 'date_x'], axis=1)
text.to_csv("Datas/tweet.txt",header=False, index=False, sep=",")
#retweetのラベルのみ取り出す
tweet_labe = data_s.drop(['tweetID', 'date_x', 'tweet2'], axis=1)
tweet_labe.to_csv("Datas/label.txt", header=False, index=False, sep=",")
