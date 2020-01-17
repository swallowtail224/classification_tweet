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
import seaborn as sns
import matplotlib as mpl
# %matplotlib inline
plt.rcParams["font.family"] = "IPAPGothic"


# +
#データ読み込み関数
def read_data(data_path):
    data_name = pd.read_csv(filepath_or_buffer=data_path, encoding="utf_8", sep=",")
    #data_name.info()
    return data_name

#各種データの変換
def conversion_id(df):
    #user_idの固有データの取り出し
    nu = df['user_id'].unique()
    #データフレームからuser_idを抽出
    u_ids = df['user_id']
    #0~31の数値へ変換
    a_post_user = []
    for i in u_ids:
        for j in range(len(nu)):
            if i == nu[j]:
                a_post_user.append(j)
                
    s = pd.DataFrame(a_post_user, columns=['user_id2'])
    df = df.join(s)
    return df

#タグをあるかないかの2値に変換
def check_tag(x):
    if x is not "0":
        return 1
    else:
        return 0
    
#タグの変換
def conversion_tag(df):
    df['tag2'] = df['tag'].apply(check_tag)
    return df

#日付の変換
def conversion_date(df):
    df['postdate'] = pd.to_datetime(df['postdate'], format = '%Y-%m-%d')
    df['date'] = df['postdate'].dt.dayofyear
    return df

#いらないデータの削減
def delete_columns(df):
    df.drop(['screen_name','user_id', 'tweet_id', 'tweet', 'tweet2', 'postdate',
               'cos_day', 'sin_day', 'tag', 'image_url'], axis = 1, inplace=True)
    data = df.loc[:, ['user_id2', 'date', 'tag2', 'image', 'retweet']]
    return data

#データの前処理
def conversion_data(data_path):
    df =  read_data(data_path)
    df = conversion_id(df)
    df = conversion_tag(df)
    df = conversion_date(df)
    return df
    
#相関係数の計算とヒートマップの作成
def calculation_corr(data):
    corr_data = data.corr()
    hmap = sns.heatmap(corr_data, vmax=1, vmin=-1, center=0, annot=True)
    return corr_data, hmap


# -

#データを読み込んで前処理
use_data = conversion_data("Datas/pickup_data.csv")
all_data = conversion_data("Datas/all_data/extract_allData.csv")
best_data = conversion_data("Datas/best_data.csv")
#相関係数の計算に使わないカラムを削除
u_data = delete_columns(use_data)
a_data = delete_columns(all_data)
b_data = delete_columns(best_data)

corr_u_data, hu_data = calculation_corr(u_data)
corr_u_data

# +
current_dpi = mpl.rcParams['figure.dpi']
print(current_dpi)

plt.figure()
sns.heatmap(corr_u_data, vmax=1, vmin=-1, center=0, annot=True)
plt.savefig('heatmap.png', dpi=current_dpi * 1.5)
# -

corr_a_data, ha_data = calculation_corr(a_data)
corr_a_data

corr_b_data, hb_data = calculation_corr(b_data)
corr_b_data

cross = pd.crosstab(use_data.screen_name,use_data.retweet)
print(cross)

cross = pd.crosstab(use_data.postdate,use_data.retweet)
print(cross[754:1119])

use_data['screen_name'].value_counts().plot.bar()

a_cross = pd.crosstab(all_data.postdate,all_data.retweet)

print(a_cross[762:1127])

print(a_cross[279:1062])


