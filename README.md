# reddit-mental-health
This repository contains the methods for producing language features from subreddits. The code was developed as part of the following paper:

> George Gkotsis, Anika Oellrich, Tim Hubbard, Richard Dobson, Maria Liakata, Sumithra Velupillai and Rina Dutta. The language of mental health problems in social media, Computational Linguistics and Clinical Psychology (clpsych, NAACL 2016).


The repository includes two Pandas Dataframes that are a small subset of the original datasets used in our study. The data provided here are mostly for demonstration purposes.

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

## Tagclouds
Follow the [link](tagclouds)