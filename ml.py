
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import content

print """USAGE:
get data as:
[['text', 'class']]

pipeline = bagOfWords()
pipeline = evaluate(pipeline, data=data)
"""

def getData(df1, df2):
    import readability
    df1 = readability.prepare(df1)
    df2 = readability.prepare(df2)

    if 'text' not in df1.columns:
        df1['text'] = df1.apply(content.getTextFromRecord, axis=1)
    if 'text' not in df2.columns:
        df2['text'] = df2.apply(content.getTextFromRecord, axis=1)
    df1['class'] = 'class1'
    df2['class'] = 'class2'

    df1 = df1[['text','class']]
    df2 = df2[['text','class']]
    data = pd.concat([df1, df2])
    data = data.reset_index()
    del data['index']
    data = data.reindex(numpy.random.permutation(data.index))
    return data

def bagOfWords(ngram_range=(2,2)):
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(ngram_range=ngram_range,analyzer='word')),
        ('classifier',         MultinomialNB())
    ])
    return pipeline

def evaluate(pipeline, data=None, getScore=False):
    k_fold = KFold(n=len(data), n_folds=6)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    precision = 0
    recall = 0
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values.astype(str)

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values.astype(str)

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='class1')
        precision += precision_score(test_y, predictions, pos_label='class1')
        recall += recall_score(test_y, predictions, pos_label='class1')
        scores.append(score)

    print('Total documents classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Precision:', precision/len(scores))
    print('Recall:', recall/len(scores))
    print('Confusion matrix:')
    print(confusion)
    if getScore:
        return pipeline, sum(scores)/float(len(scores))
    return pipeline