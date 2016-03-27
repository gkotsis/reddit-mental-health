#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy
import datetime
import nltk

#http://nbviewer.ipython.org/urls/raw.github.com/herrfz/dataanalysis/master/week3/exploratory_graphs.ipynb
#http://pandas.pydata.org/pandas-docs/version/0.9.1/visualization.html

def getCorpusDict(df):
	from nltk import tokenize
	from nltk import FreqDist
	fdist = buildCorpus(df)
	total = fdist.N()

	d = {}
	for key in fdist.iterkeys():
		d[key] = fdist[key]/float(total)

	return d

# def wordsPerSentence(post):
# 	import nltk.tokenize
# 	if len(post)==0:
# 		return 0
# 	tokenized_sentences=nltk.sent_tokenize(post)
# 	wordsCount = 0
# 	for each_sentence in tokenized_sentences:
# 		words=nltk.tokenize.word_tokenize(each_sentence)
# 		wordsCount += len(words)

# 	length = len(post)
# 	wordsPerSentence = wordsCount/float(len(tokenized_sentences))
# 	return wordsPerSentence

# def longestSentence(post):
# 	import nltk.tokenize
# 	if len(post)==0:
# 		return 0
# 	tokenized_sentences=nltk.sent_tokenize(post)
# 	maximumWordsInSentece = 0
# 	for each_sentence in tokenized_sentences:
# 		words=nltk.tokenize.word_tokenize(each_sentence)
# 		if len(words)>maximumWordsInSentece:
# 			maximumWordsInSentece = len(words)

# 	if not type(maximumWordsInSentece) is int:
# 		print post
# 	return maximumWordsInSentece

# def charactersPerWord(post):
# 	import nltk.tokenize
# 	if len(post)==0:
# 		return 0
# 	tokenized_sentences=nltk.sent_tokenize(post)
# 	wordsCount = 0
# 	characterCount = 0
# 	for each_sentence in tokenized_sentences:
# 		words=nltk.tokenize.word_tokenize(each_sentence)
# 		wordsCount += len(words)
# 		for w in words:
# 			characterCount += len(w)

# 	length = len(post)
# 	charactersPerWord = characterCount / float(wordsCount)
# 	return charactersPerWord

# def getTime():
# 	import datetime
# 	current_time = datetime.datetime.now()
# 	rs = str(current_time.day)+"/"+str(current_time.month)+"/"+str(current_time.year)+"--"+str(current_time.time().hour) + ":" + str(current_time.time().minute) + ":" + str(current_time.time().second)
# 	return rs

# def log(message):
# 	import sys
# 	rs = "[" + getTime() + "]\t" + message
# 	print rs
# 	sys.stdout.flush()
# 	return rs

def prepare(df):
	posts = df[df['parent_id'].astype(str)=='nan']
	# posts = posts[posts['selftext'].str.len()>1]
	posts = posts[posts['selftext']!='[deleted]']
	comments = df[df['parent_id'].astype(str)!='nan']
	# comments = comments[comments['body'].str.len()>1]
	comments = comments[comments['body']!='[deleted]']
	df = pd.concat([posts, comments])
	return df

# def updateBaselineFeatures(df):
# 	log("starting updateBaselineFeatures...")
# 	df['charactersPerWord'] = df['Body'].map(charactersPerWord)
# 	log("finished characters per word")
# 	df['wordsPerSentence'] = df['Body'].map(wordsPerSentence)
# 	log("finished words per sentence")
# 	df['length'] = df['Body'].map(len)
# 	log("finished length")
# 	df['longestSentence'] = df['Body'].map(longestSentence)
# 	log("finished longest sentence")
# 	return df

def updateVocabularyFeatures(df):
	import content
	log("starting updateVocabularyFeatures...")
	if 'text' not in df.columns:
		df['text'] = df.apply(content.getTextFromRecord, axis=1)
	fdist = buildCorpus(df)
	log("finished fdist")
	df['LL'] = df['text'].apply(postLikehood,args=(fdist,))
	log("finished log likehood")
	return df

# def updateAll(df):
# 	df = prepare(df)
# 	df = updateBaselineFeatures(df)
# 	df = updateVocabularyFeatures(df)
# 	df = updatePopularity(df)
# 	df = updateAnswerAge(df)
# 	df = updateNumberOfAnswers(df)
# 	df = updateScoreRatio(df)
# 	return df

# def clean(post):
# 	import string
# 	# print post
# 	if type(post) is float:
# 		return post
# 	post = post.strip()
# 	post = nltk.clean_html(post)
# 	for punct in string.punctuation:
# 		post = post.replace(punct, " ")
# 	post = post.strip()
# 	return post

# def getStats(df):
# 	import tagsExtraction as te
# 	rs = {}
# 	# df = updateReadability(df)
# 	replies = len(df[df.ParentId>0])
# 	posts = len(df[df.ParentId==0])
# 	rs['Average replies'] = df.AnswerCount.mean()
# 	rs['Longest Thread'] = df.AnswerCount.max()
# 	df['TagCount'] = df.Tags.map(te.tagCount)
# 	rs['Avg tags'] = df.TagCount.mean()
# 	rs['Answer/tags corr'] = df.AnswerCount.corr(df.TagCount)
# 	return rs

# def printCorrelations(df):
# 	from scipy.stats.stats import pearsonr
# 	print "titleReadability <-> ViewCount"
# 	print pearsonr(df.titleReadability, df.ViewCount)
# 	print "bodyReadability <-> ViewCount"
# 	print pearsonr(df.bodyReadability, df.ViewCount)
# 	print "charactersPerWord <-> ViewCount"
# 	print pearsonr(df.charactersPerWord, df.ViewCount)
# 	print "wordsPerSentence <-> ViewCount"
# 	print pearsonr(df.wordsPerSentence,df.ViewCount)
# 	print "length <-> ViewCount"
# 	print pearsonr(df.length,df.ViewCount)
# 	print "longestSentence <-> ViewCount"
# 	print pearsonr(df.longestSentence,df.ViewCount)
# 	return

# def getAcceptedAnswers(df):
# 	accepted = pd.DataFrame(df[df.AcceptedAnswerId>0].AcceptedAnswerId)
# 	accepted = pd.merge(df, accepted, left_on='Id', right_on='AcceptedAnswerId')
# 	del accepted['AcceptedAnswerId_x']
# 	del accepted['AcceptedAnswerId_y']
# 	return accepted

# def getNonAcceptedAnswers(df):
# 	answers = df[df.ParentId>0].Id
# 	answers = set(answers.tolist())
# 	accepted = set(df[df.AcceptedAnswerId>0].AcceptedAnswerId.tolist())
# 	nonAcceptedSet = answers.difference(accepted)
# 	nonAcceptedList = []
# 	for s in nonAcceptedSet:
# 		nonAcceptedList.append(s)
# 	nonAcceptedDF = pd.DataFrame(nonAcceptedList)
# 	nonAcceptedDF.rename(columns={0:'Id'}, inplace=True)
# 	rs = pd.merge(df, nonAcceptedDF, left_on='Id', right_on='Id')
# 	return rs

# def getAnswers(df):
# 	answers = df[df.ParentId>0]
# 	return answers

# def updatePopularity(df):
# 	today = df.CreationDate.max()
# 	log("starting updatePopularity...")
# 	df['popularity'] = today
# 	df['popularity'] = df['popularity'] - df.CreationDate
# 	df['popularity'] = df['popularity'].apply(lambda x:x/numpy.timedelta64(1,'D'))
# 	df.popularity = df.popularity.astype(float)
# 	df.popularity = df.ViewCount/df.popularity
# 	return df

def buildCorpus(df):
	from nltk import tokenize
	from nltk import FreqDist

	body = " ".join(df.dropna(subset=['text'])['text'].tolist())
	words = nltk.tokenize.word_tokenize(body)
	fdist=FreqDist(words)
	return fdist

def postLikehood(post, fdist):
	from nltk import tokenize
	from nltk import FreqDist
	import math
	if post is None:
		return numpy.NAN
	if post=='[deleted]':
		return numpy.NAN
	words = nltk.tokenize.word_tokenize(post)
	if len(words)==0:
		return numpy.NAN
	fpost = FreqDist(words)
	rs = 0
	for w in fpost.keys():
		# print w +"\t"+ str(math.log( fdist[w]/float(fdist.N())   , 2))
		if fdist[w]>0:
			rs += fpost[w] * math.log( fdist[w]/float(fdist.N())   , 2)

	rs = rs/len(words)
	return rs

# def updateAnswerAge(df):
# 	import numpy as np
# 	log("starting updateAnswerAge...")
# 	if 'age' in df.columns:
# 		del df['age']
# 	questions = getQuestions(df)
# 	questions = questions[['Id', 'CreationDate']]
# 	df = pd.merge(df, questions, left_on='ParentId',right_on='Id', copy=False)
# 	del df['Id_y']
# 	df.rename(columns={'Id_x':'Id'}, inplace=True)
# 	df.rename(columns={'CreationDate_x':'CreationDate'}, inplace=True)
# 	df.rename(columns={'CreationDate_y':'age'}, inplace=True)
# 	df.age = df.CreationDate - df.age
# 	df.age = df.age.astype(int)/float(86400000000000)
# 	questions['age'] = float(0)
# 	df = df.append(questions)
# 	df.ParentId = df.ParentId.replace(np.NaN,0)
# 	return df

# def updateNumberOfAnswers(df):
# 	import numpy as np
# 	log("starting updateNumberOfAnswers...")
# 	df.AnswerCount = df[df.ParentId>0].groupby('ParentId')['AnswerCount'].transform(len)
# 	return df

# def updateColumnPosition(df, column,asc=False):
# 	tmp2 = df.sort(['ParentId',column], ascending=[True,False]).reset_index(drop=True)
# 	tmp2 = tmp2.reset_index()
# 	grouped = tmp2.groupby('ParentId')
# 	print "finished grouping"
# 	UserColumnRank = pd.Series()
# 	i = 0
# 	for k,gp in grouped:
# 		i +=1
# 		if i % 10000 ==0:
# 			p = (100*i)/float(len(grouped))
# 			print i, " of ", len(grouped), p, "%"
# 		UserColumnRank = UserColumnRank.append(gp.sort(column,ascending=asc).reset_index(drop=True).Id)
# 	columnName = column+"Rank"
# 	UserColumnRank = UserColumnRank.reset_index()
# 	UserColumnRank.rename(columns={0:'Id'}, inplace=True)
# 	UserColumnRank.rename(columns={'index':columnName}, inplace=True)
# 	df = pd.merge(df, UserColumnRank,left_on='Id', right_on='Id')
# 	df[columnName] += 1
# 	return df

# def updateScoreRatio(df):
# 	import numpy as np
# 	log("starting updateScoreRatio...")
# 	df['ScoreRatio'] = 0
# 	df['ScoreRatio'] = df[df.ParentId>0].groupby('ParentId')['Score'].transform(np.sum)
# 	df.ScoreRatio = df.ScoreRatio.astype(float)
# 	df.ScoreRatio = df.Score/df.ScoreRatio

# 	return df

# def getQuestions(df):
# 	questions = df[df.ParentId<=0]
# 	return questions

# def getActivity(df):
# 	grouped = df.groupby(df.CreationDate.map(lambda t: str(t.year)+str(t.month).zfill(2))).size()
# 	return grouped

# print "prepare(df), updateBaselineFeatures(df), updateVocabularyFeatures(df), updateAnswerAge(df) or updateAll(df) is all you need"
