# reddit-mental-health
This repository contains the methods for producing language features from subreddits. If you use the code and want to cite our work, please use the following paper:

> George Gkotsis, Anika Oellrich, Tim Hubbard, Richard Dobson, Maria Liakata, Sumithra Velupillai and Rina Dutta. The Language of Mental Health Problems in Social Media, Computational Linguistics and Clinical Psychology (clpsych, NAACL 2016).

[paper](CLPsych10.pdf)

[supplement](CLPsych10_OptionalAttachment.pdf)

The repository includes two Pandas Dataframes that are a small subset of the original datasets used in our study. The data provided here are mostly for demonstration purposes.

The complete dataset we used can be found in reddit ([comments](https://redd.it/3bxlg7), [posts](https://redd.it/3mg812)).

## Installation

Follow [requirements.txt](requirements.txt)
(spaCy has an extra [step](https://spacy.io/docs#install))

## Language features

For the syntactic features, run:

```python
import pandas as pd
import content
df = pd.read_pickle("suicidewatch-sample.pickle")
df = content.addSyntacticFeatures(df)
```

For the affection features, run:
```python
import afinnsenti
import labmt
df['text'] = df.apply(content.getTextFromRecord, axis=1)
df = afinnsenti.addEmotionalFeature(df)
df = labmt.addEmotionalFeature(df)
```




## Binary classification
```python
import binaryClassification
binaryClassification.main()
rs = binaryClassification.readResults()
```
The complete output of the classification results is also stored as a dictionary in pickle format (file: *combinations-10fold.pickle*)

## Wordclouds
Follow the [link](wordclouds)