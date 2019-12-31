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
import urllib.request
import unicodedata


# +
#テキストのクリーニング
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w])+', '', text_string)
    text_string = re.sub(r'[0-9]', '', text_string)
    text_string = re.sub(r'[０-９]', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

#slothlibのストップワードの取得
def get_stopword():
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
    return slothlib_stopwords

#ストップワードの追加
def add_stopword(slothlib_stopwords):
    s = open("Datas/stopwords.txt", "r", encoding="utf-8")
    stop = s.readlines()
    s.close()
    stop = [a.strip() for a in stop]
    slothlib_stopwords += stop
    return slothlib_stopwords

#MeCabによる形態素解析
def extractter(text, flag):
    text = unicodedata.normalize("NFKC", text)
    tagger = mc.Tagger(r'-Ochasen -d G:\neologd')
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
            if flag == 0:
                if nes[0] == u"名詞":
                    if nes[6] == u"*":
                         keyword.append(wor)
                    else:
                        keyword.append(nes[6])
            else:
                if nes[0] == u"名詞":
                    if nes[6] == u"*":
                         keyword.append(wor)
                    else:
                        keyword.append(nes[6])
                elif nes[0] == u"動詞":
                    if nes[6] == u"*":
                        keyword.append(wor)
                    else:
                        keyword.append(nes[6])
                elif nes[0] == u"形容詞":
                    if nes[6] == u"*":
                        keyword.append(wor)
                    else:
                        keyword.append(nes[6])
                elif nes[0] == u"副詞":
                    if nes[6] == u"*":
                        keyword.append(wor)
                    else:
                        keyword.append(nes[6])
        node = node.next
        if node is None:
            break
    return keyword

#ストップワードの除去
def except_stopwords(text, stopwords):
    for i in text:
        for j in stopwords:
            if i == j:
                text.remove(j)
    return text


# +
#メイン
#ツイートのテキスト読み込み
test = open("Datas/tweet.txt", "r", encoding="utf-8")
lines = test.readlines()
test.close()
print(len(lines))

#クリーニング及び形態素解析
text_data = [clean_text(x) for x in lines]
M_extract_text = [extractter(y, 0) for y in text_data]
A_extract_text = [extractter(z, 1) for z in text_data]
stopwords = get_stopword()
ex_stopwords = add_stopword(stopwords)
M_result = [except_stopwords(a, ex_stopwords) for a in M_extract_text]
A_result = [except_stopwords(b, ex_stopwords) for b in A_extract_text]
M_result = [' '.join(d) for d in M_result]
A_result = [' '.join(d) for d in A_result]
# -

#書き出し1
g = open("Datas/N_extract_tweet.txt", "w", encoding='utf-8')
for i in M_result:
    g.write(i)
    g.write('\n')
g.close()

#書き出し2
h = open("Datas/A_extract_tweet.txt", "w", encoding='utf-8')
for i in A_result:
    h.write(i)
    h.write('\n')
h.close()
