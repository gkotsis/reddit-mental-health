#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import nltk


def getCorpusDict(df):
    from nltk import tokenize
    from nltk import FreqDist
    fdist = buildCorpus(df)
    total = fdist.N()

    d = {}
    for key in fdist.keys():
        d[key] = fdist[key] / float(total)

    return d


def prepare(df):
    posts = df[df['parent_id'].astype(str) == 'nan']
    # posts = posts[posts['selftext'].str.len()>1]
    posts = posts[posts['selftext'] != '[deleted]']
    comments = df[df['parent_id'].astype(str) != 'nan']
    # comments = comments[comments['body'].str.len()>1]
    comments = comments[comments['body'] != '[deleted]']
    df = pd.concat([posts, comments])
    return df


def updateVocabularyFeatures(df):
    import content
    log("starting updateVocabularyFeatures...")
    if 'text' not in df.columns:
        df['text'] = df.apply(content.getTextFromRecord, axis=1)
    fdist = buildCorpus(df)
    log("finished fdist")
    df['LL'] = df['text'].apply(postLikelihood, args=(fdist,))
    log("finished log likelihood")
    return df


def buildCorpus(df):
    from nltk import tokenize
    from nltk import FreqDist

    body = " ".join(df.dropna(subset=['text'])['text'].tolist())
    words = nltk.tokenize.word_tokenize(body)
    fdist = FreqDist(words)
    return fdist


def postLikelihood(post, fdist):
    from nltk import tokenize
    from nltk import FreqDist
    import math
    if post is None:
        return np.NAN
    if post == '[deleted]':
        return np.NAN
    words = nltk.tokenize.word_tokenize(post)
    if len(words) == 0:
        return np.NAN
    fpost = FreqDist(words)
    rs = 0
    for w in fpost.keys():
        # print w +"\t"+ str(math.log( fdist[w]/float(fdist.N())   , 2))
        if fdist[w] > 0:
            rs += fpost[w] * math.log(fdist[w] / float(fdist.N()), 2)

    rs = rs / len(words)
    return rs
