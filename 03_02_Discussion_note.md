# 🌋 3rd meeting : Natural Language Processing with Disaster Tweets 

**written by Soyoun Son**         
**Date : 050922**

#### 🦆 Kaggle : https://www.kaggle.com/c/nlp-getting-started

## 🌱 풀잎스쿨 (5월 9일) 

서로 같이 논의된 부분 및 공부한 것에 대해서 적음

### ☺︎ attention 
: 디코더에서 출력 단어를 예측하는 매 시점마다 인코더에서의 전체입력 문장을 다시 한번 참고.
- dot production attention 
- bert-> transformer-> attention 

<img src="/images/attention.png" width="500">
ref fig : https://github.com/jadore801120/attention-is-all-you-need-pytorch


대부분의 문제를 푸는데, electra 또는 bert임 
Bert-using-TFHUB.ipynb

### ☺︎ Bert
ref: https://wikidocs.net/115055

- google이 공개한 사전훈련된 모델 
- 33억개로 사전 훈련된 모델 
- input 
   + subword tokenizer : wordpiece라는 토크나이저 사용함 
   + positional embedding : 단어의 위치 정보를 찾기 위해서 사용하는 것 (단어의 위치를 알수 없으므로) 
   + attention mask  
  : 이 3개가 합쳐져서 bert의 input으로 들어감 
- bert structure 
   + multihead self attention : 512가 맥스
     (self-attention : 단어 유사도를 확인할 수 있음)
     https://wikidocs.net/89786
         
   + rasidual connection  
   + normalization 
   + feed forward NN 

- Bert train 
  + MLM
  + NSP : 문장분류, binary classification 

ref : https://www.kaggle.com/code/wrrosa/keras-bert-using-tfhub-modified-train-data


### ☺︎ Ensemble 
ensemble 의 경우 TF-IDF를 이용한 후 사용함 




---------------
encoding은 줄이는 것이고, GPT말고 구글이 내놓은 것이 PaLM (줄이는 것이 아니라, 메모리를 늘리는 것)
+ google : PaLM
http://aidev.co.kr/chatbotdeeplearning/11284
https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html

+ meta : OPT 

+ Alpha code : https://towardsdatascience.com/deepminds-alphacode-explained-everything-you-need-to-know-5a86a15e1ab4


embedded : 800K
domain : 


개발적으로보면 (프로그램안에서의) 아키텍쳐 바꾼다고 성능이 좋아진다고 볼수 없다. -> 유지/보수
(시스템안에서의) 아키텍쳐는 성능이 개선 가능  
최적화는 버린다는 것의 의미


-----------------
ML 개념? 
call back function : 도중에 끊을수가 없으니깐, listener같은 것  
+ early stopping
+ ModelCheckpoint
+ ReduceLROnPlateau
+ csvLogger




Solve NLP proble, with deep learning : https://www.kaggle.com/code/megr25/twitter-nlp-feature-engineer-deep-learning

