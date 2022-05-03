# Natural Language Processing with Disaster Tweets 

🦆 https://www.kaggle.com/c/nlp-getting-started

> problem statement : You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
(즉, 트윗 문장을 보고 disaster 나타내는 문장인지, 그와 관련된 단어가 있는 문장인지 판단하는 문제) [1,2]

- [x] Dataset
- [ ] EDA 
- [ ] Methodologies

#### ☺︎ Dataset 
데이터셋에 대한 이해가 필요함 

#### ☻ Data Files [1]
+ train.csv : the training set
+ test.csv : the test set
+ sample_submission.csv : a sample submission file in the correct format

#### ☺︎ dataset [1]
데이터 셋의 columns은 아래와 같고, 

+ id : a unique identifier for each tweet
+ text : the text of the tweet
+ location : the location the tweet was sent from (may be blank)
+ keyword : a particular keyword from the tweet (may be blank)
+ target : in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

각 파일별 차이가 존재하는데 test.csv의 경우 4개의 columns이 존재하고, train.csv의 경우 label (target) 까지 총 5개의 columns이 존재

#### ☺︎ EDA [3]
EDA는 target이 1과 0으로 나눠서 data distribution 및 visualization을 진행하였다. 

+ number of characters, words in tweets
+ average word lenght in a tweet
+ common stopwords in tweets
+ analyzing punctuations 
+ common words? : 어떤 단어들이 
+ common stop
+ common stopwords in tweetsㅁ
+ common stopwords in twe



#### ☺︎ Methodologies




### ref 
[1] https://www.kaggle.com/competitions/nlp-getting-started/data
[2] https://onground-korea.github.io/machine_learning/2021/03/07/Kaggle-NLP.html
[3] https://www.kaggle.com/code/shahules/basic-eda-cleaning-and-glove
