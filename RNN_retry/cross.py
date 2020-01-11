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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams["font.family"] = "IPAPGothic"

#データの読み込み
use_data = pd.read_csv(filepath_or_buffer="Datas/pickup_data.csv", encoding="utf_8", sep=",")
print(len(use_data))
use_data.info()

# +
#user_idの固有データの取り出し
nu = use_data['user_id'].unique()
print(nu)
#データフレームからuser_idを抽出
ids = use_data['user_id']
#0~31の数値へ変換
post_user = []
for i in ids:
    for j in range(len(nu)):
        if i == nu[j]:
            post_user.append(j)

#データフレームへ変換
s = pd.DataFrame(post_user, columns=['user_id2'])
use_data = use_data.join(s)


# +
#タグをあるかないかの2値に変換
def conversion_tag(x):
    if x is not "0":
        return 1
    else:
        return 0
    
use_data['tag2'] = use_data['tag'].apply(conversion_tag)
use_data.head()
# -

use_data['postdate'] = pd.to_datetime(use_data['postdate'], format = '%Y-%m-%d')
use_data['date'] = use_data['postdate'].dt.dayofyear

use_data.drop(['screen_name','user_id', 'tweet_id', 'tweet', 'tweet2', 'postdate',
               'cos_day', 'sin_day', 'tag', 'image_url'], axis = 1, inplace=True)

use_data

datas = use_data.loc[:, ['user_id2', 'date', 'tag2', 'image', 'retweet']]

data = datas.corr()

data

# +
import seaborn as sns

sns.heatmap(data, vmax=1, vmin=-1, center=0, annot=True)
# -

cross = pd.crosstab(use_data.screen_name,use_data.retweet)
print(cross)

cross = pd.crosstab(use_data.postdate,use_data.retweet)
print(cross[754:1119])

use_data['screen_name'].value_counts().plot.bar()


