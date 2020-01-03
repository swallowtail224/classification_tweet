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

#tweet dataの読み込み
tweet_input = pd.read_csv(filepath_or_buffer="Datas/all_data/tweet_datas.csv", encoding="utf_8", sep=",")
print(len(tweet_input))
#メディアURLの読み込み
image_input = pd.read_csv(filepath_or_buffer="Datas/all_data/t_mediaurl.csv", encoding="utf_8", sep=",")
print(len(image_input))
#ハッシュタグの読み込み
tag_input = pd.read_csv(filepath_or_buffer="Datas/all_data/t_hashtag_data.csv", encoding="utf_8", sep=",") 
print(len(tag_input))

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

tweet_data.shape

#ハッシュタグと画像のまとめ上げ
tags = tag_input.groupby("tweet_id").agg({'tag': lambda x: ','.join(x)})
print(tags.shape)
images = image_input.groupby("tweet_id").agg({'image_url': lambda x: ','.join(x)})
print(images.shape)

tweet_data.head()

#不要な文字の削除
tweet_data["tweet"] = tweet_data["tweet"].str.replace('\n', ' ').str.replace('_nan', ' ').str.replace(r'[0-9]', ' ').str.replace(r'([^\s\w])+', ' ').str.replace(r'[０-９]', ' ')

tweet_data.head()

datas = pd.merge(tweet_data, tags, on='tweet_id', how='left')
all_data = pd.merge(datas, images, on='tweet_id', how='left')
all_data.head(10)

#NaNデータのカウント
print(all_data.isnull().sum())

all_data = all_data.fillna(0)
print(all_data.isnull().sum())
all_data.head(10)

all_data.to_csv("Datas/all_data/all_data.csv",index=False, sep=",")

#特定のデータ数のカウント
test = ((tweet_data['screen_name'] == '川源ぶどう園(花巻市)') & (tweet_data['retweet'] == 1.0)) 
print(test.sum())
