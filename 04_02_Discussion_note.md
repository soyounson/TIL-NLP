# π¦ Quora Question Pairs (QQP)

**written by Soyoun Son**         
**Date : 051722**


#### π¦ Kaggle ref : https://www.kaggle.com/c/quora-question-pairs

## π± νμμ€μΏ¨ (5μ 17μΌ 2022λ, λͺ¨λμ°κ΅¬μ μΊ νΌμ€) 

![Fig00](/image/google_bert.png)


## π»ββοΈ λ°ν 1 
- baseline training μ΄ν μ±λ₯μ λμ΄κΈ° μν΄μ EDA μμ μ§ν 
- **[lime](https://github.com/marcotcr/lime) library : DLμμ μ λλ‘ νμ΅νκ³  μλμ§λ₯Ό NLPμ μ μ©ν¨ **
(ref: https://c3.ai/glossary/data-science/lime-local-interpretable-model-agnostic-explanations/) 
(ref2 : https://christophm.github.io/interpretable-ml-book/lime.html)
 
## π»ββοΈ λ°ν 2 
### βΊοΈ λ¨μ΄νν
μ½νΌμ€ κ΅¬μ±νκ³  id μ£Όλ κ² 
### ν΅κ³κΈ°λ° κΈ°λ² : λ¨μ΄ λΉλμμ λ°λΌ λ²‘ν°λ‘ νν TF-IDF, μ½μ¬μΈ μ μ¬λ, 
μ λ³μνΈμ λ³΄λ (pointwise mutual information) 
### μΆλ‘ κΈ°λ°κΈ°λ² : λΆν¬κ°μ€μ κΈ°μ΄λ₯Ό νμ§λ§ λ¬Έμ₯λ΄μμ μ§μ½μ μΈ μμ­, CBOW, skip-gram-> λκ° λ¬Άμ΄μ word2vec

- CBOW 
- word2vec : λ§λ­μΉκ³μκ° λ§μμ§λ©΄ λ³΅μ‘λκ° μ»€μ§ -> μλκ°μ  λ°©λ²μΌλ‘ embedding, negative sampling (sigmoid fnμ΄μ©ν΄μ μ΄μ§λΆλ₯) 

### βΊοΈ [GloVe : Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
ν΅κ³κΈ°λ°ννλ°©μ + μλ―ΈμΆλ‘  
extract global corpus statistics information
λμλ°μνλ₯ μ λν λΉμ¨μ κ΅¬νλ κ²μ΄ λ μ€μν¨μ λνλ. λΉμ¨μ λ°λΌμ μΈμ§νμλ μ’ λ κ°μμ μΌλ‘ λνλ΄κΈ° λλ¬Έμ 
μλ² λ©λ μ€μ¬λ¨μ΄μ μ£Όλ³λ¨μ΄ λ²‘ν°μ λ΄μ μ΄ μ μ²΄ μ½νΌμ€μμμ λμ λ±μ₯νλ₯ μ΄ λλλ‘ λ§λλ κ² 
μ λ³΄λμ λ°λΌμ κ°μ€μΉλ₯Ό μ€ 

### βΊοΈ embedding layer 
μ΄μ°μ μΈ κ°λ±μ μμΉμ μΈ κ°μΌλ‘ λ§΅νμν€λ κ² 
https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer/notebook
[next sentend prediction, NSP](https://towardsdatascience.com/bert-for-next-sentence-prediction-466b67f8226f)

### Bert μ λ¨μ΄ μλ² λ© 
#### input representation 
- token embedding 
- segment embedding 
- position embedding 
-> λͺ¨λ κ°μ€μΉ λ²‘ν°λ₯Ό κ°κ³ μμ κ³μ° 

#### λ΄λΆμ½λ
tf.gradients()λ‘ νμ΅ 


## π»ββοΈ λ°ν 3, [NLP λ¬Έμ ν΄κ²°μ λ΅](https://www.notion.so/modulabs/NLP-bdc7562bc0e146c69cbf55cf9590dcf7)
#### process 
understand problem -> EDA -> baseline models -> improve performance 

### μμ°μ΄ μ²λ¦¬ λ¬Έμ 
- λ¬Έμ₯μ μ΄ν΄νλ©΄ νμμλ λ¬Έμ λ€μ΄ μλΉμλ₯Ό μ°¨μ§ν¨. κ·Έλμ μ΄κ²μΈμ§λ₯Ό νμΈνκ³  λ² μ΄μ€λΌμΈμΌλ‘ κ°. 
- μμΉνλ‘ λνλ΄μΌν¨
- λ¬Έμ₯μ μ΄ν΄νλ©΄ νμμλ λ¬Έμ λ€μ΄ μλΉμλ₯Ό μ°¨μ§ν¨. κ·Έλμ μ΄κ²μΈμ§λ₯Ό νμΈνκ³  λ² μ΄μ€λΌμΈμΌλ‘ κ°. 
- κ·Έλ¦¬κ³  λμ λ² μ΄μ€λΌμΈλλΉ μ΄λ€μμΌλ‘ μ΅μ ν ν κ²μΈμ§ μκ°ν΄ λ³Ό κ² 
- μ΅μ νλ λμ²΄μ μΌλ‘ ν¨ν΄μ΄ μμ. 
- out of the box (creative) μΈ κ²½μ°λ λΌλ¬Έμ μμ±νλ κ² (1λμ΄μ μκ°μμ) λ°λΌμ μΊκΈκ³Όλ κ±°λ¦¬κ° μμ. 
- NLU : κΈλ£¨μ 8κ° μνΌκΈλ£¨μ 9κ°? μ΄λ°μμΌλ‘?
 - [wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) : λͺ¨λ  λ¨μ΄λ€μ μ νν μμΌλμ 
- NLG : λ²μ­, λ¬Έμ₯μ λ§λ€μ΄ λ΄λ κ² 
  - QQP : μ μ¬λμμΈ‘ λ¬Έμ μ ν΄λΉν¨ 

- [papers w/ code](https://paperswithcode.com/)
#### GLUE 
9κ°μ μ νμΌλ‘ λ§λ€μ΄ λμ 
#### Super GLUE
- κΈ°μλ€μ΄ ν¨κ» λ§λ  κ². 8κ°μ μ νμΌλ‘ λ§λ€μ΄ λμ 
- κ΅¬λΆμ λ°μ΄ν°μμ μΆμ², ν¬κΈ°, λΌλ²¨λ±μ λ°λΌμ λμ΄ μμ

### μμ°μ΄μ κ²½μ°
1. μ§μμ΄μ© -> domain knowledgeκ° νμ
2. μλ―Έμ΄ν΄μ΄μ© 
3. μλ² λ© μμ€ : λ¬Έμκ° λ€μ΄μ€λ©΄ κ·Έκ²μ΄ λ²‘ν°κ° λλ κ³Όμ μ΄λ κ²°κ³Όλ¬Ό 
  (λ¨μ΄-> λ¬Έμ₯ -> λ¬Έμ), pretrained modelμμ bertκ° λμ΄ 
  μ΄κ²μ μ΄λ ν νμμΌλ‘ λνλΈ κ² [EEAP](https://explosion.ai/blog/deep-learning-formula-nlp#embed)
  NLPλͺ¨λΈμ λν SOTAκ³΅μ
  κ°μ₯ dependentν κ²μ΄ ν ν¬λμ΄μ ? ν ν¬λμ΄μ κ° λ¨Όμ  νμ΅μ νκ³ 
  0μ΄ μλ κ²μΌλ‘ ν ν¬λμ΄μ  νλ κ²μΌλ‘μ¨ λ²νΈκ° ?
  a. μλ² λ© : fixed sizeκ° ν€ μ. νμ¬μμ€μμλ μλ² λ©μ΄ λ€ νμνλ€.
  b. μΈμ½λ 
[Roberta: a robustly optimized bert pretraining approach](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md) 
LSTMμ λ£λ κ²μ΄ νλμ νμΌλ‘ λ³Ό μ μμ. 

![Fig01](/image/Untitled.png)

 c. attend : dimensionμ΄ νλ μ€μ΄μΌ ν¨. clsν΅ν΄μ λ½λ κ² λ³΄λ€ λ λμ μ μλ₯Ό μ€μλ μμ. 
 ![Fig01](/image/bert_attention.png)
 d. predict: FFNN -> relu, gelu, tanh -> normalization -> dropout (classificationλ£κΈ° λ°λ‘ μ μ)
    κ·Έλ¦¬κ³  [optuna](https://optuna.org/)κ° μ°Ύλλ‘ ν¨ 
    λ¨μ΄λ€κ³Ό λ¨μ΄λ€ μ¬μ΄μ idμ λ³΄, positionν μ λ³΄κ° κ°μ΄ λ€μ΄κ° 
    
#### out of the box : someth creative  
papers w/ code    
μλ¨μ λ°κΎΈλ©΄ μ’μ κ²°κ³Όκ° λμ¬μκ° μλ€. νΉμ λ§μ³μ§κ±°λ?!
T5 : encoder-decoder 
spanbert : ν ν°μ μ΄μ΄ λΆμΈ λ¦¬μ€νΈ? ν ν°? μ΄κ²μ λ§μ€ν¬λ‘ μ¬μ©ν¨ 
deberta : bertμ νμμΈλ° 3κ°μ§μ λμ ν° μμ΄λμ΄λ₯Ό λ£μ΄μ μκ³ λ¦¬μ¦μ μΌλ‘ ν΄κ²°ν¨ 
token : λ¨μ΄μ§ μ΅μ λ¨μ
μ€μ  λΉμ§λμ€μμλ νμΈνλ

#### extra 
CLS: stands for classification. It is added at the beginning because the training tasks here is sentence classification. And because they need an input that can represent the meaning of the entire sentence, they introduce a new tag.

#### νμΈν΄λ³Ό λ΄μ© 
LIME, OPTUNA

#### Bert

![Fig02](/image/bert-sentence-pair.png)
sentence 1, sentence 2 
sepμ μ΄λ―Έ μμ½λμ΄ μλ κ² 

MT-DNN κ΄λ ¨ λΌλ¬Έ : https://y-rok.github.io/nlp/2019/05/20/mt-dnn.html

bertκ° 512κ°λ‘ μ νλμ΄ μμΌλ―λ‘ 

####
bertλλ¬Έμ νμ νλ‘μ°μμ νμ΄ν μΉλ‘ μ΄λ (λ­κ° μ’ λ dynamics) 
κ·Έλμ νμνλ‘μ°2κ° λμ΄
κ·Όλ°, νμ΄ν μΉμ μ»€λ?€λν°κ° λ νΌ 


=====================================

### data imbalance problem 
- λ°μν μ μλ λ¬Έμ  : κ³Όμ ν© 
- undersampling : data loss > **λ΄μΌ μ μ©ν΄ λ³Ό κ²** 
- oversampling : data generation (most used method) 

ref : https://joonable.tistory.com/27
ref : https://xper100.tistory.com/7



### logloss μμ€ν¨μ 
Log Loss is the most important classification metric based on probabilities. Itβs hard to interpret raw log-loss values, but log-loss is still a good metric for comparing models. For any given problem, a lower log loss value means better predictions.

Mathematical interpretation:
Log Loss is the negative average of the log of corrected predicted probabilities for each instance.
Let us understand it with an example

[ref] https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/

#### λ€μμ£Ό 
papers w code 
[leaderboard](https://www.kaggle.com/competitions/quora-question-pairs/leaderboard) 

λ¦¬λλ³΄λμ μλ λͺ¨λΈλ€μ λ³΄κ³ , μ΄ λͺ¨λΈλ€μ μ΄λ€ νΉμ±λλ¬Έμ μ΄ λͺ¨λΈμ μ±λ₯μ΄ μ’μμ§ λ³΄κ³ , 
μ΄κ²μ λμ€μ μ΄λ»κ² λμ€μ μ¬μ©ν  μ μμμ§μ λν΄μ μ λ¦¬ν¨. 
https://paperswithcode.com/dataset/quora-question-pairs
