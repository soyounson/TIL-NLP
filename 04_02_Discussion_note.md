# 🦖 Quora Question Pairs (QQP)

**written by Soyoun Son**         
**Date : 051722**


#### 🦆 Kaggle ref : https://www.kaggle.com/c/quora-question-pairs

## 🌱 풀잎스쿨 (5월 17일 2022년, 모두연구소 캠퍼스) 


## 🐻‍❄️ 발표 1 
- baseline training 이후 성능을 높이기 위해서 EDA 작업 진행 
- [lime](https://github.com/marcotcr/lime) library : DL에서 제대로 학습하고 있는지를 NLP에 적용함 
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


## 🐻‍❄️ 발표, [NLP 문제해결전략](https://www.notion.so/modulabs/NLP-bdc7562bc0e146c69cbf55cf9590dcf7)
#### process 
understand problem -> EDA -> baseline models -> improve performance 

### 자연어 처리 문제
- 문장을 이해하면 풀수있는 문제들이 상당수를 차지함. 그래서 이것인지를 확인하고 베이스라인으로 감. 
- 그리고 나서 베이스라인대비 어떤식으로 최적화 할것인지 생각해 볼 것 
- 최적화는 대체적으로 패턴이 있음. 
- out of the box (creative) 인 경우는 논문을 작성하는 것 (1년이상 시간소요) 따라서 캐글과는 거리가 있음. 
- NLU : 글루에 8개 슈퍼글루에 9개? 이런식으로?
- NLG : 
- [wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) : 모든 단어들을 정형화 시켜놓음 












