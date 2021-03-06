# ๐ Natural Language Processing with Disaster Tweets 

**written by Soyoun Son**         
**Date : 050222**


#### ๐ฆ Kaggle ref : https://www.kaggle.com/c/nlp-getting-started

> + problem statement : You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.     
+ ๋ฌธ์ : ์ฆ, ํธ์ ๋ฌธ์ฅ์ ๋ณด๊ณ  disaster ๋ํ๋ด๋ ๋ฌธ์ฅ์ธ์ง, ๊ทธ์ ๊ด๋ จ๋ ๋จ์ด๊ฐ ์๋ ๋ฌธ์ฅ์ธ์ง ํ๋จํ๋ ๋ฌธ์  [1,2]


### โบ๏ธ Process
- [x] Dataset
- [x] EDA 
- [x] Methodologies

### โบ๏ธ Dataset 
๋ฐ์ดํฐ์์ ๋ํ ์ดํด๊ฐ ํ์ํจ 

#### โป Data Files [1]
+ train.csv : the training set
+ test.csv : the test set
+ sample_submission.csv : a sample submission file in the correct format

#### โป dataset [1]
๋ฐ์ดํฐ ์์ columns์ ์๋์ ๊ฐ๊ณ , 

+ id : a unique identifier for each tweet
+ text : the text of the tweet
+ location : the location the tweet was sent from (may be blank)
+ keyword : a particular keyword from the tweet (may be blank)
+ target : in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

๊ฐ ํ์ผ๋ณ ์ฐจ์ด๊ฐ ์กด์ฌํ๋๋ฐ test.csv์ ๊ฒฝ์ฐ 4๊ฐ์ columns์ด ์กด์ฌํ๊ณ , train.csv์ ๊ฒฝ์ฐ label (target) ๊น์ง ์ด 5๊ฐ์ columns์ด ์กด์ฌ

> id, keyword, text ๊ฐ ์ค์ํ ๊ฒ 

#### โป Exploratory Data Analysis, EDA [3]
EDA๋ target์ด 1๊ณผ 0์ผ๋ก ๋๋ ์ data distribution ๋ฐ visualization์ ์งํํ์๋ค. 

+ number of characters, words in tweets
+ average word lenght in a tweet
+ common stopwords in tweets
+ analyzing punctuations 
+ common words? : ์ด๋ค ๋จ์ด๋ค์ด ๋ง์ด ์ฌ์ฉ๋์๋์ง ํ์ธ 
+ Ngram analysis : do a bigram (n=2) analysis over the tweets
```
vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
bag_of_words = vec.transform(corpus)
```
#### โป Data cleaning 
basic cleaning such as **spelling correction,removing punctuations,removing html tags and emojis etc.***
+ removing urls
+ removing HTML tags
+ removing Emojis
+ removing punctuations
+ spelling correction 

#### โป GloVe for Vecttorization [4]
use GloVe pretrained corpus model to represent our words.It is available in 3 varieties 

๊ธ๋ก๋ธ(Global Vectors for Word Representation, GloVe)๋ ์นด์ดํธ ๊ธฐ๋ฐ๊ณผ ์์ธก ๊ธฐ๋ฐ์ ๋ชจ๋ ์ฌ์ฉํ๋ ๋ฐฉ๋ฒ๋ก ์ผ๋ก 2014๋์ ๋ฏธ๊ตญ ์คํ ํฌ๋๋ํ์์ ๊ฐ๋ฐํ ๋จ์ด ์๋ฒ ๋ฉ ๋ฐฉ๋ฒ๋ก ์๋๋ค. ์์ ํ์ตํ์๋ ๊ธฐ์กด์ ์นด์ดํธ ๊ธฐ๋ฐ์ LSA(Latent Semantic Analysis)์ ์์ธก ๊ธฐ๋ฐ์ Word2Vec์ ๋จ์ ์ ์ง์ ํ๋ฉฐ ์ด๋ฅผ ๋ณด์ํ๋ค๋ ๋ชฉ์ ์ผ๋ก ๋์๊ณ , ์ค์ ๋ก๋ Word2Vec๋งํผ ๋ฐ์ด๋ ์ฑ๋ฅ์ ๋ณด์ฌ์ค๋๋ค. ํ์ฌ๊น์ง์ ์ฐ๊ตฌ์ ๋ฐ๋ฅด๋ฉด ๋จ์ ์ ์ผ๋ก Word2Vec์ GloVe ์ค์์ ์ด๋ค ๊ฒ์ด ๋ ๋ฐ์ด๋๋ค๊ณ  ๋งํ  ์๋ ์๊ณ , ์ด ๋ ๊ฐ์ง ์ ๋ถ๋ฅผ ์ฌ์ฉํด๋ณด๊ณ  ์ฑ๋ฅ์ด ๋ ์ข์ ๊ฒ์ ์ฌ์ฉํ๋ ๊ฒ์ด ๋ฐ๋์งํฉ๋๋ค.

### โบ๏ธ Baseline Methodologies

embedding + LSTM ๋ชจ๋ธ ๋ง๋ค๊ณ ,  
```
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

optimzer=Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
```
๋ฐ์ดํฐ๋ฅผ train/test๋ก ๋๋๊ณ , training ๋ฐ prediction๊น์ง ์งํํจ. 

```
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
y_pre=model.predict(test)
```


## โป Reference
[1] https://www.kaggle.com/competitions/nlp-getting-started/data         
[2] https://onground-korea.github.io/machine_learning/2021/03/07/Kaggle-NLP.html            
[3] https://www.kaggle.com/code/shahules/basic-eda-cleaning-and-glove            
[4] https://wikidocs.net/22885
