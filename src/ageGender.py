from spacy.lang.en import English
import pandas as pd
import os

# download the lexicon files from http://wwbp.org/data.html
# add files emnlp14age.csv and emnlp14gender.csv in the current directory
# see paper http://aclweb.org/anthology/D/D14/D14-1121.pdf for implementation details

if 'SPACY_DATA' in os.environ:
    data_dir = os.environ['SPACY_DATA']
else:
    data_dir = None
print("Load EN from %s" % data_dir)

nlp = English(data_dir=data_dir)

ageL = pd.read_csv("emnlp14age.csv")
ageL = ageL.set_index('term')
ageL = ageL['weight'].to_dict()
ageL_intercept = ageL['_intercept']
genderL = pd.read_csv("emnlp14gender.csv")
genderL = genderL.set_index('term')
genderL = genderL['weight'].to_dict()
genderL_intercept = genderL['_intercept']


def stripMD(text):
    from bs4 import BeautifulSoup
    from markdown import markdown

    text = markdown(text)
    text = ''.join(BeautifulSoup(
        text, features="html.parser").findAll(text=True))
    return text


def getValue(text, t='Age'):
    if text is None:
        return None
    text = stripMD(text)
    text = text.lower()
    if t == 'Age':
        lexicon = ageL
        intercept = ageL_intercept
    else:
        lexicon = genderL
        intercept = genderL_intercept
    from collections import Counter
    tokens = list(nlp.tokenizer(str(text)))
    tokens = [token.text for token in tokens]
    d = Counter(tokens)
    # only keep the words that are in the dictionary
    d = {key: value for key, value in d.items() if key in lexicon.keys()}
    total_words = sum(d.values())
    value = intercept
    for word in d.keys():
        if word in lexicon.keys():
            weight = lexicon[word]
            x = d[word] / float(total_words)
            value += weight * x
    return value


def getAge(text):
    rs = getValue(text, t='Age')
    return rs


def getGender(text):
    rs = getValue(text, t='Gender')
    print(rs)
    if rs < 0:
        return "M"
    return "F"
