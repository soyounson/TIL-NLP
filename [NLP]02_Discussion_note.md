# Natural Language Processing with Disaster Tweets 

>written by Soyoun Son\
>Date : 050222


🦆 https://www.kaggle.com/c/nlp-getting-started

> problem statement : You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.\ 문제: 즉, 트윗 문장을 보고 disaster 나타내는 문장인지, 그와 관련된 단어가 있는 문장인지 판단하는 문제 [1,2]

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

> id, keyword, text 가 중요한 것 

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

--------------------------------------------------------------
## 🌱 풀잎스쿨 (5월 2일) 

서로 같이 논의된 부분 및 공부한 것에 대해서 적음


[NLP preprocess](https://www.kaggle.com/code/longtng/nlp-preprocessing-feature-extraction-methods-a-z/notebook) 

1. read and explore data 
2. text cleaning 
capitalization : 대문자-> 소문자 
expand the contractions
remove noise, punctuations
특수문자는 character로 바꿈 
한국어로 잡업할때는 capitalization, expand the contractions부분이 필요하지 않음 
3. text pre-processing 
+ tokenization 
+ remove stop words : nltk에 tokenizer 사전이 존재. 한국어의 경우는 POS tagging 이후에 진행
+ stemming : 어간추출, 분석결과가 안좋았을때 하는 경우가 일반적 
+ POS tagging : 품사를 지정하는 것. 동사/명사/... 문맥적으로 잘 파악할수 있도록 하는 과정. 품사에 따라서 의미가 달라지는 경우가 발생. 한국어의 경우 KoNLPy
+ Lemmatization : 표제어 추출. 기본형으로 바꿔주는 과정. 따라서 품사 지정하는 PoS tagging한 후에 진행해야 함. 
+ (optional) language detection 

한국어의 경우 토큰화-> POS tagging -> stemming/lemmatization -> remove stop words
한국어의 경우 표제어 추출, 어간추출의 차이가 많이 없고, 어간추출이 그나마 더 많이 사용됨

4. Text Feature Extraction 
(1) weighted words - BOW
- countvectorizer
- TF-IDF
카운팅하지만, 위치정보를 잃음. 그래서 의미적 유사성 파악 불가 
(2) word embedding 
- word2vec   -> gensim 
- Glove
- FastText
- Bert : transformer 30억개 이상의 단어를 미리 학습시킨후, 본인의 테스크에 따라서 fine tunning을 하면 정확도가 높음. 
(3) comparison of feature extraction techinique  

-> 어떤 의미있는 feature을 뽑아낼수 있을까? 

예로 Bert의 한계가 존재한다면, 

이런 문제였을때는 다른 것 사용 할 것
다른 접근법: 각 방법의 약점을 확인하면 좋을 듯
Bert (Transformer)만 사용하다보니, 기존 모델들 RNN
A : architecture 


++++++++++

[TF-IDF 개념](https://www.notion.so/modulabs/NLP-w-DL-061fbb36c67d494fa062309914b4842d?p=f0a9b205d22545fe9e0707b8493f57ac) 
(1) 전처리 작업
- 긍정문서/부정문서
- tf-idf 
- 등장 키워드에 대한 scoring 
(2) 모델 
- decision tree, regression 
실무 적용시 : 재난의 유무가 아니라 재난의 카테고리까지 찾아야 한다.


Questions 

실무에 적용한다면, 
- 긍정/부정끼리 모으는 전처리 : 사람이 annotation이 되야함.
- TF-IDF만 사용한다면 2번을 태워야 할 것
- 키워드 지식 : 사전구축, 실제 자연어처리를 할때, 특정 도메인에 맞는 단어에 가중치를 줌. 은행 관련 도메인의 키워드는 '계좌이체'이런 것들. 
키워드 지식이 이미 갖춰진 pre-trained model이 있다면 

공공도메인 데이터의 경우, 표준화가 된 데이터임. 
실제 데이터를 받았을때는 줄이는 것이 필요. 

TF-IDF대신 countvector이용해서 TF만 사용하는 것이 더 나을듯 
Bert에 최적화되어 있는 것 

++++++++++

[10 steps for challenging and improving NLP model](https://www.notion.so/modulabs/NLP-w-DL-061fbb36c67d494fa062309914b4842d?p=bc0c6c7c725246f9a2920c87fb2a6af3)

3. one-hot encoding : GPT같은 것은 라벨이 onehot encoding으로 나옴. 
4. 길이가 들쭉날쭉한것 보다 패딩있는 것이 결과가 좋았다?
5. 커브가 내려가다가 뛰면 오버피팅하는 것이고, 계속 내려가면 상관없음 
shuffle : 데이터가 뭔가 연속적으로 비슷한 형태라면 그것을 막기위해서 섞는 것
파이토치의 경우 첫번째 에포크에서 이미 결정이 난다. 
10. scale factors 

****** 실제 실무에서 확인 할 것 ***
모델 바꾸는 것보다 데이터에서 승부가 나는 것 
차라리 토크너나이저같은 것을 바꾸는 것이 중요함 
역삼각형이어서 모델을 바꾸는 것보다, 초기를 바꾸는 것이 좋음 
버트도 버트의 모델보다 oov줄여서 성능이 좋아지는 것
토크나이저에서 워드 피스를 줄이는 것. 
파싱할때 (한국어에서 어간, 어미 쪼갤때)와 같은 토크나이저를 썼을때 성능 올라감
여전히 도메인이 중요. 도메인에서 많이 쓰는 언어가 단어로 할당됨?


### kaggle notebook을 github으로 갖고 오는 경우 
https://www.kaggle.com/product-feedback/295170

https://somjang.tistory.com/category/Kaggle/Real%20or%20Not%3F%20NLP%20with%20Disaster%20Tweets


목적 최적화된 모델 찾는것?
Gridserch CV로 hyperparameter?
전처리 작업을 진행안하고
전처리작업하고 진행하는 것이 아니라, 전처리를 뒤에서 확인하는 것?
비교하는 것은 accuracy?
score? 20%에 어떤것들이 틀렸는지? 그것을 확인하는 것?
모델에 따라서 못 맞추는 것들이 다르지 않을까? 
epoch 1-5 : 성능은 앞단에서 결정된다는 것을 기반으로 생각한건가? 

hyperparatmer 튜닝을 AutoML이 다해줌?

더 좋은 성능을 갖고 있다고 해서 좋은 모델이라고 볼수 있는가? 
모집단에서 샘플을 뽑아서 확인할때, 그 샘플이 모집단을 그대로 반영하고 있는 것일까? 

좋은 성적 얻기 위한 모델 [ref](https://github.com/MLFS19-NLP/KaggleNotebooks/blob/main/nlp-a-gentle-introduction-lstm-word2vec-bert.ipynb)
LSTM, simple RNN, word2Vec w/ Gensim, BERT,Voting
-> 오답을 봐서 (점수가 비슷하더라도) 

오답이 몰려잇다하면 합치면
엑스쥐부스트 : 오답을 줄이는 모델, 대신 과적합 잘됨.

LSTM : 맨 뒤에
BERT, simpleRNN : 맨앞에
Glove : 합쳐서 가능?

-------------
engineering : 모델을 적용해서 데이터를 분석하고 고쳐가는 방법 

***자연어 처리에 대해서 내부적인 main features***
1. TF 
2. 단어와의 거리 
3. 단어의 순서/방향

마스크 모델? 

***문제 해결 방법***

1. 읽어보고, 라벨링이 가능한가? 언어를 적용할수 있는지?
2. Bert로 가는지? 아님 구조적으로 가야하는지? 
승부는 전처리에서 나온다.
Hyperparameter : AutoML, optuna 
약어같은 경우는 어떤식으로? Bert에서 잘못인식할수 있을듯
단어를 겹치게 하면 
마킹? 마스킹을 걸어놓음 
베이스라인을 하고 오답을 보면서 맞춰가면서 이해함. 

Bert의 토큰제한화 


-------

우승자 방법에 대한 설명 
우리가 언급했던 부분에 대해서 각자 돌려보고 발표 


