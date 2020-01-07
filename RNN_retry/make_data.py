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

#データの読み込み
all_data = pd.read_csv(filepath_or_buffer="Datas/all_data/A_extract_allData.csv", encoding="utf_8", sep=",")
print(len(all_data))
all_data.info()

#NaNデータのカウント
print(all_data.isnull().sum())
#NaNのデータを削除
use_data = all_data.dropna(how='any')
#掲載したツイート数のカウント
published_post = use_data['retweet'] == 1
published_post.sum()

#掲載したツイートとそうでなかったツイートで分割
publish_tweet = use_data.query('retweet == 1')
non_publish_tweet = use_data.query('retweet == 0')
#それぞれから1万件ランダム抽出
p_post = publish_tweet.sample(n=10000, random_state=1)
n_post = non_publish_tweet.sample(n=10000, random_state=1)
#抽出したデータを結合
data = pd.concat([p_post,n_post])
pickup_data = data.sort_index()

pickup_data.to_csv("Datas/A_pickup_data.csv",index=False, sep=",")

#確認
test = pd.read_csv(filepath_or_buffer="Datas/A_pickup_data.csv", encoding="utf_8", sep=",")
test.info()


