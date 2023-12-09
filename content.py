from spacy.lang.en import English
import pandas as pd
import numpy as np
import os
PoS = ['NN', 'PRP', 'IN', 'DT', 'RB', 'JJ', 'VB', 'CC', 'NNS', 'VBP']
FirstPersonPronouns = ['i', 'me', 'mine', 'my',
                       'myself', 'our', 'ours', 'ourselves', 'us', 'we']

if 'SPACY_DATA' in os.environ:
    data_dir = os.environ['SPACY_DATA']
else:
    data_dir = None
print("Load EN from %s" % data_dir)

nlp = English(data_dir=data_dir)


def getTextFromRecord(row):
    if not pd.isnull(row['body']):
        rs = row['body']
    else:
        rs = row['selftext']
    if rs == '[deleted]':
        return None
    return rs


def addLexicalFeatures(df):
    from collections import Counter
    df['text'] = df.apply(getTextFromRecord, axis=1)

    def cosineSimilarity(sentence1, sentence2, NounsOrPronounsOnly=False):
        import math
        if not NounsOrPronounsOnly:
            vector1 = Counter([token.text.lower() for token in sentence1])
            vector2 = Counter([token.text.lower() for token in sentence2])
        else:
            vector1 = Counter([token.text.lower() for token in sentence1 if token.tag_.startswith(
                "N") or token.tag_.startswith("PR")])
            vector2 = Counter([token.text.lower() for token in sentence2 if token.tag_.startswith(
                "N") or token.tag_.startswith("PR")])
        intersection = set(vector1.keys()) & set(vector2.keys())
        numerator = sum([vector1[x] * vector2[x] for x in intersection])
        sum1 = sum([vector1[x] ** 2 for x in vector1.keys()])
        sum2 = sum([vector2[x] ** 2 for x in vector2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def getDocumentSimilarity(text, NounsOrPronounsOnly=False):
        doc = nlp(str(text))
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return None
        rs = 0
        for i in range(0, len(sentences) - 1):
            rs += cosineSimilarity(sentences[i],
                                   sentences[i + 1], NounsOrPronounsOnly)
        rs = rs / float(len(sentences) - 1)
        return rs

    def getPronounsCounter(text):
        doc = nlp(str(text))
        sentences = list(doc.sents)
        from collections import Counter
        pronouns = []
        for sentence in sentences:
            pronouns.extend(
                [token.text for token in sentence if token.tag_.startswith('PRP')])

        pronouns = Counter(pronouns)
        pronounsNo = np.sum(list(pronouns.values()))
        return pd.Series({'sentencesNo': len(sentences), 'pronouns': pronouns, 'pronounsNo': pronounsNo})

    def getDefiniteArticlesCounter(text):
        doc = nlp(str(text))
        sentences = list(doc.sents)
        from collections import Counter
        definiteArticles = []
        for sentence in sentences:
            definiteArticles.extend([token.text for token in sentence if (
                token.tag_ == 'DT' and token.text.lower() == 'the')])

        definiteArticles = Counter(definiteArticles)
        definiteArticlesNo = np.sum(list(definiteArticles.values()))
        return pd.Series({'definiteArticlesNo': definiteArticlesNo})

    def getFirstPersonPronounsCounter(counter):
        l = [k for k in counter.keys() if k.lower() in FirstPersonPronouns]
        rs = {}
        for k in l:
            rs[k] = counter[k]
        firstPersonPronouns = Counter(rs)
        firstPersonPronounsNo = np.sum(list(firstPersonPronouns.values()))
        return pd.Series({'firstPersonPronouns': firstPersonPronouns, 'firstPersonPronounsNo': firstPersonPronounsNo})

    df['documentSimilarity'] = df['text'].apply(getDocumentSimilarity)
    df['documentSimilarityNounsOrPronouns'] = df['text'].apply(
        getDocumentSimilarity, args=(True,))
    df[['pronouns', 'pronounsNo', 'sentencesNo']
       ] = df['text'].apply(getPronounsCounter)
    df[['definiteArticlesNo']] = df['text'].apply(getDefiniteArticlesCounter)
    df[['firstPersonPronouns', 'firstPersonPronounsNo']
       ] = df['pronouns'].apply(getFirstPersonPronounsCounter)
    df['firstPersonPronounsRatio'] = df['firstPersonPronounsNo'] / \
        df['pronounsNo'].astype(float)
    return df


def getSyntacticFeatures(row):
    text = getTextFromRecord(row)

    def getHeightToken(token):
        height = 1
        while token != token.head:
            height += 1
            token = token.head
        return height

    def getVerbPhrasesLength(sentence):
        rs = [0]
        inVerb = False
        for token in sentence:
            if token.pos_ == 'VERB':
                if not inVerb:
                    i = 1
                inVerb = True
                i += 1
            if token.pos_ != 'VERB':
                if inVerb:
                    rs.append(i - 1)
                inVerb = False
        return rs

    if text is None:
        return pd.Series({'maxHeight': np.nan, 'noun_chunks': np.nan, 'maxVerbPhraseLength': np.nan,
                          'subordinateConjuctions': np.nan}, dtype=object)
    doc = nlp(str(text))
    noun_chunks = len(list(doc.noun_chunks))
    sentences = list(doc.sents)
    maxHeight = 1

    subordinateConjuctions = 0
    for sentence in sentences:
        subordinateConjuctions += len(
            [token for token in sentence if token.tag_ == 'IN'])
    subordinateConjuctions = subordinateConjuctions / float(len(sentences))

    for sentence in sentences:
        height = 1
        if sentence.end == 0:
            continue
        sentenceHeight = max([getHeightToken(token) for token in sentence])
        if maxHeight < sentenceHeight:
            maxHeight = sentenceHeight

    maxVerbPhraseLength = max(
        max([getVerbPhrasesLength(sentence) for sentence in sentences]))
    return pd.Series({'maxHeight': maxHeight - 1, 'noun_chunks': noun_chunks,
                      'maxVerbPhraseLength': maxVerbPhraseLength, 'subordinateConjuctions': subordinateConjuctions},
                     dtype=object)


def getURLtoPostRatio(df):
    def containedInURL(row):
        return row['permalink'] not in row['url']

    posts = df[df['parent_id'].astype(str) == 'nan']
    cnt = len(posts)
    urls = len(posts[posts.apply(containedInURL, axis=1)])
    return urls / float(cnt)
