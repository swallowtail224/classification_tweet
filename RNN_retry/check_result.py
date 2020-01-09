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

use_data = pd.read_csv(filepath_or_buffer="Datas/test_data_A.csv", encoding="utf_8", sep=",")
print(len(use_data))
use_data.info()

A_result = pd.read_csv(filepath_or_buffer="Datas/result/model1_dA_predict.csv", encoding="utf_8", sep=",")
print(len(A_result))
A_result.info()

A_result

use_data.drop(['user_id','tweet_id','cos_day', 'sin_day','tag', 'image_url', 'image'], axis=1, inplace=True)

result_data = use_data.join(A_result)

result_data

published = result_data[result_data['retweet'] == 1.0]
not_published = result_data[result_data['retweet'] == 0.0]

published.to_csv("Datas/published_data_A.csv",index=False, sep=",")

not_published.to_csv("Datas/non_published_data_A.csv",index=False, sep=",")


