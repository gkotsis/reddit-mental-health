import math
import re
import sys

# AFINN-111 is as of June 2011 the most recent version of AFINN
filenameAFINN = 'AFINN/AFINN-111.txt'
afinn = dict(map(lambda ws: (ws[0], int(ws[1])), [
             ws.strip().split('\t') for ws in open(filenameAFINN)]))

# Word splitter pattern
pattern_split = re.compile(r"\W+")


def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value is negative valence.
    """
    words = pattern_split.split(text.lower())
    sentiments = map(lambda word: afinn.get(word, 0), words)
    if sentiments:
        # How should you weight the individual word sentiments?
        # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
        sentiment = float(sum(sentiments)) / math.sqrt(len(sentiments))
    else:
        sentiment = 0
    return sentiment


def score(text):
    if text is None:
        return None
    return sentiment(text)


def addEmotionalFeature(df):
    df['afinn'] = df['text'].apply(score)
    return df
