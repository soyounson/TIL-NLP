# Natural Language Processing with Disaster Tweets 

🦆 https://www.kaggle.com/c/nlp-getting-started

> problem statement : You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
(즉, 트윗 문장을 보고 disaster 나타내는 문장인지, 그와 관련된 단어가 있는 문장인지 판단하는 문제) [1,2]

- [x] Dataset
- [x] EDA 
- [x] Methodologies

### ☺︎ Dataset 
데이터셋에 대한 이해가 필요함 

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

#### ☻ Exploratory Data Analysis, EDA [3]
EDA는 target이 1과 0으로 나눠서 data distribution 및 visualization을 진행하였다. 

+ number of characters, words in tweets
+ average word lenght in a tweet
+ common stopwords in tweets
+ analyzing punctuations 
+ common words? : 어떤 단어들이 많이 사용되었는지 확인 
+ Ngram analysis : do a bigram (n=2) analysis over the tweets
```
vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
bag_of_words = vec.transform(corpus)
```
#### ☻ Data cleaning 
basic cleaning such as **spelling correction,removing punctuations,removing html tags and emojis etc.***
+ removing urls
+ removing HTML tags
+ removing Emojis
+ removing punctuations
+ spelling correction 

#### ☻ GloVe for Vecttorization [4]
use GloVe pretrained corpus model to represent our words.It is available in 3 varieties 

글로브(Global Vectors for Word Representation, GloVe)는 카운트 기반과 예측 기반을 모두 사용하는 방법론으로 2014년에 미국 스탠포드대학에서 개발한 단어 임베딩 방법론입니다. 앞서 학습하였던 기존의 카운트 기반의 LSA(Latent Semantic Analysis)와 예측 기반의 Word2Vec의 단점을 지적하며 이를 보완한다는 목적으로 나왔고, 실제로도 Word2Vec만큼 뛰어난 성능을 보여줍니다. 현재까지의 연구에 따르면 단정적으로 Word2Vec와 GloVe 중에서 어떤 것이 더 뛰어나다고 말할 수는 없고, 이 두 가지 전부를 사용해보고 성능이 더 좋은 것을 사용하는 것이 바람직합니다.

### ☺︎ Baseline Methodologies

embedding + LSTM 모델 만들고,  
```
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

optimzer=Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
```
데이터를 train/test로 나누고, training 및 prediction까지 진행함. 

```
X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)

history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
y_pre=model.predict(test)
```

### ref 
[1] https://www.kaggle.com/competitions/nlp-getting-started/data

[2] https://onground-korea.github.io/machine_learning/2021/03/07/Kaggle-NLP.html

[3] https://www.kaggle.com/code/shahules/basic-eda-cleaning-and-glove

[4] https://wikidocs.net/22885






-----------------------------------------------------------
음성인식 legacy model 
음성 신호를 fft로 변환하고 mel spectrogram 으로 image 뽑은 다음에 그 이미지로 감성분류
nlp에서 음성인식은 작은 분파이고, signal processing 이면 음성인식쪽이 더 맞음
NLP는 model architecture 쪽에 더 관심이 많음
ASR (Automated Speech Recognition) 

HMM (Hidden Markov Model) +GMM (Gaussian Mixture Model) -> RNN+GMM -> RNN -> Transformer -> Wav2Vec
RNN-> Transformer 

http://speech.cbnu.ac.kr/

NLP : ELMO -> BERT -> GPT2 -> GPT3
