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

### ☺︎ Dataset 

preprocessing + feature engineering 

#### ☻ Train dataset : train.csv 

+ id: Looks like a simple rowID
+ qid{1, 2}: The unique ID of each question in the pair
+ question{1, 2}: The actual textual contents of the questions.
+ is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other.

train 데이터 분석 결과은 아래와 같다. 

```
- Total number of question pairs for training: 404290
- Duplicate pairs: 36.92%
- Total number of questions in the training data: 537933
- Number of questions that appear multiple times: 111780
```

metrics : LogLoss looks at the actual predicts as opposed to the order of predictions, we should be able to get a decent score by creating a submission predicting the mean value of the label.


#### ☻ Test dataset : test.csv 

+ test_id: L
+ question{1, 2}: The actual textual contents of the questions.

test 데이터에는 `is_duplicate`,  즉 label이 존재하지 않음 

We see a similar distribution for word count, with most questions being about 10 words long. It looks to me like the distribution of the training set seems more "pointy", while on the test set it is wider. Nevertheless, they are quite similar.


#### ☻ Semantic Analysis
take a look at usage of different punctuation in questions - this may form a basis for some interesting features later on.

```
Questions with question marks: 99.87%
Questions with [math] tags: 0.12%
Questions with full stops: 6.31%
Questions with capitalised first letters: 99.81%
Questions with capital letters: 99.95%
Questions with numbers: 11.83%
```

### ☺︎ Initial Feature Analysis
Before we create a model, we should take a look at how powerful some features are. I will start off with the word share feature from the benchmark model.

#### ☻ TF-IDF
This means that we weigh the terms by how uncommon they are, meaning that we care more about rare words existing in both questions than common one. This makes sense, as for example we care more about whether the word "exercise" appears in both than the word "and" - as uncommon words will be more indicative of the content.


#### ☻ Rebalancing the data 
However, before I do this, I would like to rebalance the data that XGBoost receives, since we have 37% positive class in our training data, and only 17% in the test data. By re-balancing the data so our training set has 17% positives, we can ensure that XGBoost outputs probabilities that will better match the data on the leaderboard, and should get a better score (since LogLoss looks at the probabilities themselves and not just the order of the predictions like AUC)


### ☺︎ Baseline Methodologies
#### ☻ XGBoost

```
import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
```
그리고 test 작업 진행함.
```
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)
```

score is `0.35460` on the leaderboard. 



## ☻ Reference
[1] https://www.kaggle.com/competitions/nlp-getting-started/data    




