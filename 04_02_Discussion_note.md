# 🦖 Quora Question Pairs (QQP)

**written by Soyoun Son**         
**Date : 051722**


#### 🦆 Kaggle ref : https://www.kaggle.com/c/quora-question-pairs

## 🌱 풀잎스쿨 (5월 17일 2022년, 모두연구소 캠퍼스) 

![Fig00](/image/google_bert.png)


## 🐻‍❄️ 발표 1 
- baseline training 이후 성능을 높이기 위해서 EDA 작업 진행 
- **[lime](https://github.com/marcotcr/lime) library : DL에서 제대로 학습하고 있는지를 NLP에 적용함 **
(ref: https://c3.ai/glossary/data-science/lime-local-interpretable-model-agnostic-explanations/) 
(ref2 : https://christophm.github.io/interpretable-ml-book/lime.html)
 
## 🐻‍❄️ 발표 2 
### ☺︎ 단어표현
코퍼스 구성하고 id 주는 것 
### 통계기반 기법 : 단어 빈도수에 따라 벡터로 표현 TF-IDF, 코사인 유사도, 
점별상호정보량 (pointwise mutual information) 
### 추론기반기법 : 분포가설에 기초를 하지만 문장내에서 지엽적인 영역, CBOW, skip-gram-> 두개 묶어서 word2vec

- CBOW 
- word2vec : 말뭉치계수가 많아지면 복잡도가 커짐 -> 속도개선 방법으로 embedding, negative sampling (sigmoid fn이용해서 이진분류) 

### ☺︎ [GloVe : Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
통계기반표현방식 + 의미추론 
extract global corpus statistics information
동시발생확률에 대한 비율을 구하는 것이 더 중요함을 나타냄. 비율에 따라서 인지했을때 좀 더 가시적으로 나타내기 때문임 
임베딩된 중심단어와 주변단어 벡터의 내적이 전체 코퍼스에서의 동시 등장확률이 되도록 만드는 것 
정보량에 따라서 가중치를 줌 

### ☺︎ embedding layer 
이산적인 값등을 수치적인 값으로 맵핑시키는 것 
https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer/notebook
[next sentend prediction, NSP](https://towardsdatascience.com/bert-for-next-sentence-prediction-466b67f8226f)

### Bert 와 단어 임베딩 
#### input representation 
- token embedding 
- segment embedding 
- position embedding 
-> 모두 가중치 벡터를 갖고와서 계산 

#### 내부코드
tf.gradients()로 학습 


## 🐻‍❄️ 발표 3, [NLP 문제해결전략](https://www.notion.so/modulabs/NLP-bdc7562bc0e146c69cbf55cf9590dcf7)
#### process 
understand problem -> EDA -> baseline models -> improve performance 

### 자연어 처리 문제
- 문장을 이해하면 풀수있는 문제들이 상당수를 차지함. 그래서 이것인지를 확인하고 베이스라인으로 감. 
- 수치화로 나타내야함
- 문장을 이해하면 풀수있는 문제들이 상당수를 차지함. 그래서 이것인지를 확인하고 베이스라인으로 감. 
- 그리고 나서 베이스라인대비 어떤식으로 최적화 할것인지 생각해 볼 것 
- 최적화는 대체적으로 패턴이 있음. 
- out of the box (creative) 인 경우는 논문을 작성하는 것 (1년이상 시간소요) 따라서 캐글과는 거리가 있음. 
- NLU : 글루에 8개 슈퍼글루에 9개? 이런식으로?
 - [wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) : 모든 단어들을 정형화 시켜놓음 
- NLG : 번역, 문장을 만들어 내는 것 
  - QQP : 유사도예측 문제에 해당함 

- [papers w/ code](https://paperswithcode.com/)
#### GLUE 
9개의 유형으로 만들어 놓음 
#### Super GLUE
- 기업들이 함께 만든 것. 8개의 유형으로 만들어 놓음 
- 구분은 데이터셋의 출처, 크기, 라벨등에 따라서 되어 있음

### 자연어의 경우
1. 지식이용 -> domain knowledge가 필요
2. 의미이해이용 
3. 임베딩 수준 : 문자가 들어오면 그것이 벡터가 되는 과정이나 결과물 
  (단어-> 문장 -> 문서), pretrained model에서 bert가 나옴 
  이것을 어떠한 형식으로 나타낸 것 [EEAP](https://explosion.ai/blog/deep-learning-formula-nlp#embed)
  NLP모델에 대한 SOTA공식
  가장 dependent한 것이 토크나이저? 토크나이저가 먼저 학습을 하고
  0이 아닌 것으로 토크나이저 하는 것으로써 버트가 ?
  a. 임베딩 : fixed size가 키 임. 현재수준에서는 임베딩이 다 필요하다.
  b. 인코드 
[Roberta: a robustly optimized bert pretraining approach](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md) 
LSTM을 넣는 것이 하나의 팁으로 볼 수 있음. 

![Fig01](/image/Untitled.png)

 c. attend : dimension이 하나 줄어야 함. cls통해서 뽑는 것 보다 더 높은 점수를 줄수도 있음. 
 d. predict: FFNN -> relu, gelu, tanh -> normalization -> dropout (classification넣기 바로 전에)
    그리고 [optuna](https://optuna.org/)가 찾도록 함 
    단어들과 단어들 사이의 id정보, position한 정보가 같이 들어감 
    
#### out of the box : someth creative  
papers w/ code    
앞단을 바꾸면 좋은 결과가 나올수가 있다. 혹은 망쳐지거나?!
T5 : encoder-decoder 
spanbert : 토큰을 이어 붙인 리스트? 토큰? 이것을 마스크로 사용함 
deberta : bert의 후속인데 3가지정도의 큰 아이디어를 넣어서 알고리즘적으로 해결함 
token : 떨어진 최소 단위
실제 비지니스에서는 파인튜닝

#### extra 
CLS: stands for classification. It is added at the beginning because the training tasks here is sentence classification. And because they need an input that can represent the meaning of the entire sentence, they introduce a new tag.

#### 확인해볼 내용 
LIME, OPTUNA

#### Bert

![Fig02](/image/bert-sentence-pair.png)
sentence 1, sentence 2 
sep은 이미 예약되어 있는 것 
