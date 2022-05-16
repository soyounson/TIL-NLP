# 🦖 Quora Question Pairs 

**written by Soyoun Son**         
**Date : 051622**


#### 🦆 Kaggle ref : https://www.kaggle.com/c/quora-question-pairs

### ☺︎ Purpose of this competition 
Our goal is to identify which questions asked on Quora, a quasi-forum website with over 100 million visitors a month, are duplicates of questions that have already been asked. This could be useful, for example, to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not, and submitting a binary prediction against the logloss metric.

### ☺︎ 노트 정리
most voted code를 분석하고 이해하는 작업을 수행함으로써, 문제/데이터/솔루션에 대한 이해를 한다. 

### ☺︎ Contents
- [ ] Dataset
- [ ] Initial feature analysis 
- [ ] Model : Xgboost
- 
### ☺︎ Dataset 

preprocessing + feature engineering 

#### ☻ Data Files [1]
+ train.csv 
+ test.csv 

#### ☻ dataset [1]
+ id: Looks like a simple rowID
+ qid{1, 2}: The unique ID of each question in the pair
+ question{1, 2}: The actual textual contents of the questions.
+ is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

#### ☻ Text analysis


#### ☻ Semantic Analysis
Next, I will take a look at usage of different punctuation in questions - this may form a basis for some interesting features later on.


### ☺︎ Initial Feature Analysis
Before we create a model, we should take a look at how powerful some features are. I will start off with the word share feature from the benchmark model.


#### ☻ TF-IDF

#### ☻ Rebalancing the data 

### ☺︎ Baseline Methodologies
#### ☻ XGBoost
score is `0.35460` on the leaderboard. 

```


```

```

```





## ☻ Reference
[1] https://www.kaggle.com/competitions/nlp-getting-started/data    




