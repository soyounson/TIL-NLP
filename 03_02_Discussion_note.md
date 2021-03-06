# ๐ 3rd meeting : Natural Language Processing with Disaster Tweets 

**written by Soyoun Son**         
**Date : 050922**

#### ๐ฆ Kaggle : https://www.kaggle.com/c/nlp-getting-started

## ๐ฑ ํ์์ค์ฟจ (5์ 9์ผ) 

์๋ก ๊ฐ์ด ๋ผ์๋ ๋ถ๋ถ ๋ฐ ๊ณต๋ถํ ๊ฒ์ ๋ํด์ ์ ์
์ถ๊ฐ์ ์ผ๋ก ์ ๋ฆฌ๊ฐ ํ์ํจ 


๋ชจ๋ธ์ ๋๋ ค๋ณด๊ณ , ์ฑ๋ฅ๋น๊ต 

### โบ๏ธ attention 
: ๋์ฝ๋์์ ์ถ๋ ฅ ๋จ์ด๋ฅผ ์์ธกํ๋ ๋งค ์์ ๋ง๋ค ์ธ์ฝ๋์์์ ์ ์ฒด์๋ ฅ ๋ฌธ์ฅ์ ๋ค์ ํ๋ฒ ์ฐธ๊ณ .
- dot production attention 
- bert-> transformer-> attention 

<img src="/images/attention.png" width="500">
ref fig : https://github.com/jadore801120/attention-is-all-you-need-pytorch


๋๋ถ๋ถ์ ๋ฌธ์ ๋ฅผ ํธ๋๋ฐ, electra ๋๋ bert์ 
Bert-using-TFHUB.ipynb

### โบ๏ธ Bert
ref: https://wikidocs.net/115055

- google์ด ๊ณต๊ฐํ ์ฌ์ ํ๋ จ๋ ๋ชจ๋ธ 
- 33์ต๊ฐ๋ก ์ฌ์  ํ๋ จ๋ ๋ชจ๋ธ 
- input 
   + subword tokenizer : wordpiece๋ผ๋ ํ ํฌ๋์ด์  ์ฌ์ฉํจ 
   + positional embedding : ๋จ์ด์ ์์น ์ ๋ณด๋ฅผ ์ฐพ๊ธฐ ์ํด์ ์ฌ์ฉํ๋ ๊ฒ (๋จ์ด์ ์์น๋ฅผ ์์ ์์ผ๋ฏ๋ก) 
   + attention mask  
  : ์ด 3๊ฐ๊ฐ ํฉ์ณ์ ธ์ bert์ input์ผ๋ก ๋ค์ด๊ฐ 
- bert structure 
   + multihead self attention : 512๊ฐ ๋งฅ์ค
     (self-attention : ๋จ์ด ์ ์ฌ๋๋ฅผ ํ์ธํ  ์ ์์)
     https://wikidocs.net/89786
         
   + rasidual connection  
   + normalization 
   + feed forward NN 

- Bert train 
  + MLM
  + NSP : ๋ฌธ์ฅ๋ถ๋ฅ, binary classification 

ref : https://www.kaggle.com/code/wrrosa/keras-bert-using-tfhub-modified-train-data


### โบ๏ธ Ensemble 
ensemble ์ ๊ฒฝ์ฐ TF-IDF๋ฅผ ์ด์ฉํ ํ ์ฌ์ฉํจ 




---------------
encoding์ ์ค์ด๋ ๊ฒ์ด๊ณ , GPT๋ง๊ณ  ๊ตฌ๊ธ์ด ๋ด๋์ ๊ฒ์ด PaLM (์ค์ด๋ ๊ฒ์ด ์๋๋ผ, ๋ฉ๋ชจ๋ฆฌ๋ฅผ ๋๋ฆฌ๋ ๊ฒ)
+ google : PaLM
http://aidev.co.kr/chatbotdeeplearning/11284
https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html

+ meta : OPT 

+ Alpha code : https://towardsdatascience.com/deepminds-alphacode-explained-everything-you-need-to-know-5a86a15e1ab4


embedded : 800K
domain : 


๊ฐ๋ฐ์ ์ผ๋ก๋ณด๋ฉด (ํ๋ก๊ทธ๋จ์์์์) ์ํคํ์ณ ๋ฐ๊พผ๋ค๊ณ  ์ฑ๋ฅ์ด ์ข์์ง๋ค๊ณ  ๋ณผ์ ์๋ค. -> ์ ์ง/๋ณด์
(์์คํ์์์์) ์ํคํ์ณ๋ ์ฑ๋ฅ์ด ๊ฐ์  ๊ฐ๋ฅ  
์ต์ ํ๋ ๋ฒ๋ฆฐ๋ค๋ ๊ฒ์ ์๋ฏธ


-----------------
ML ๊ฐ๋? 
call back function : ๋์ค์ ๋์์๊ฐ ์์ผ๋๊น, listener๊ฐ์ ๊ฒ  
+ early stopping
+ ModelCheckpoint
+ ReduceLROnPlateau
+ csvLogger


Bert      
XLnet      
T5     
ELMO 

news ์ฐ๊ฑฐ๋ ์์์ ๋ง๋ค์ด๋ด๊ธฐ๋ ํ๋๋ฐ, ์ด๋ decoder์ ์จ์ผํ๋๋ฐ, ์ด๊ฑฐ๋ autoregressiveํ๊ณ  ์ด๊ฒ์ only unidirectionํ๋ค. 
bert์ฌ์ฉํ๋ฉด ์ฑ๋ฅ์ด ๋จ์ด์ง???   


------------------
Solve NLP proble, with deep learning : https://www.kaggle.com/code/megr25/twitter-nlp-feature-engineer-deep-learning

----------------


๋ค์์ฃผ ํด์ผ ํ  ๊ฒ 
data analysis 
EDA
baseline model 
์ด๋ก ์ ์ธ ๊ฒ ํ์ธ 
