# 🦖 Quora Question Pairs 

**written by Soyoun Son**         
**Date : 051722**


#### 🦆 Kaggle ref : https://www.kaggle.com/c/quora-question-pairs

## 🌱 풀잎스쿨 (5월 17일) 


- baseline training 이후 성능을 높이기 위해서 EDA 작업 진행 
- [lime](https://github.com/marcotcr/lime) library : DL에서 제대로 학습하고 있는지를 NLP에 적용함 
(ref: https://c3.ai/glossary/data-science/lime-local-interpretable-model-agnostic-explanations/) 
 


### ☺︎ 단어표현
코퍼스 구성하고 id 주는 것 
### 통계기반 기법 : 단어 빈도수에 따라 벡터로 표현 TF-IDF, 코사인 유사도, 
점별상호정보량 (pointwise mutual information) 
### 추론기반기법 : 분포가설에 기초를 하지만 문장내에서 지엽적인 영역, CBOW, skip-gram-> 두개 묶어서 word2vec

- CBOW 
- word2vec : 말뭉치계수가 많아지면 복잡도가 커짐
     https://wikidocs.net/89786
