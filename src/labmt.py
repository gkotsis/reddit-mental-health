import pandas as pd
import math

# url = 'http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0026752.s001'
labmt = pd.read_csv('labmt.txt', skiprows=2, sep='\t', index_col=0)

average = labmt.happiness_average.mean()
happiness = (labmt.happiness_average - average).to_dict()


def score(text):
    if text is None:
        return None
    if len(text.strip()) <= 0:
        return None
    words = text.split()
    return sum([happiness.get(word.lower(), 0.0) for word in words]) / math.sqrt(len(words))


def addEmotionalFeature(df):
    df['labmt'] = df['text'].apply(score)
    return df
