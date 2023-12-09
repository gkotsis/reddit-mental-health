# Socialmedia AI Analysis and Mental Health Predictive Intervention

# Updates

- Refactored and Upgraded to python3 by [Adam D](https://github.com/adamd1985).

# Refernce Papers and Original Authors


> George Gkotsis, Anika Oellrich, Tim Hubbard, Richard Dobson, Maria Liakata, Sumithra Velupillai and Rina Dutta. The Language of Mental Health Problems in Social Media, Computational Linguistics and Clinical Psychology ([clpsych](http://hollingk.github.io/CLPsych/index.html), NAACL 2016).

[paper](CLPsych7.pdf)
[supplement](CLPsych7_OptionalAttachment.pdf)

The complete dataset we used can be found in reddit ([comments](https://redd.it/3bxlg7), [posts](https://redd.it/3mg812)).

## Installation

Follow [requirements.txt](requirements.txt)
(spaCy has an extra [step](https://spacy.io/docs#install))

## Language features

For the syntactic features, run:

```python
import pandas as pd
import content
df = pd.read_pickle("suicidewatch-sample.pkl")
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
The complete output of the classification results is also stored as a dictionary in pickle format (file: *combinations-10fold.pkl*)

## Wordclouds
Follow the [link](wordclouds)