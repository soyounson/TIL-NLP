# 🌋 Natural Language Processing with Disaster Tweets 

**written by Soyoun Son**         
**Date : 050922**


#### 🦆 Kaggle ref : https://www.kaggle.com/c/nlp-getting-started

### ☺︎ Contents
- [] Dataset
- [] Model : **Naive Bayes Classification**

### ☺︎ Dataset 

preprocessing + feature engineering 

#### ☻ Data Files [1]
+ train.csv : the training set
+ test.csv : the test set
+ sample_submission.csv : a sample submission file in the correct format

#### ☻ dataset [1]
데이터 셋의 columns은 아래와 같고, 

+ id : a unique identifier for each tweet
+ text : the text of the tweet
+ location : the location the tweet was sent from (may be blank)
+ keyword : a particular keyword from the tweet (may be blank)
+ target : in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

각 파일별 차이가 존재하는데 test.csv의 경우 4개의 columns이 존재하고, train.csv의 경우 label (target) 까지 총 5개의 columns이 존재

> id, keyword, text 가 중요한 것 


### ☺︎ Baseline Methodologies

embedding + LSTM 모델 만들고,  
```


```
데이터를 train/test로 나누고, training 및 prediction까지 진행함. 

```
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
y_pre=model.predict(test)
```





## ☻ Reference
[1] https://www.kaggle.com/competitions/nlp-getting-started/data    




