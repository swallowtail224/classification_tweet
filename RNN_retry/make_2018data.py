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

#データの読み込み
use_data = pd.read_csv(filepath_or_buffer="Datas/all_data/extract_allData.csv", encoding="utf_8", sep=",")
print(len(use_data))
use_data.info()

data2018 = use_data[27189:49730]

data2018.to_csv("Datas/data2018.csv",index=False, sep=",")

best_data = use_data[6385:46656]

best_data.to_csv("Datas/best_data.csv",index=False, sep=",")


