110221


Lagrange multiplier (https://en.wikipedia.org/wiki/Lagrange_multiplier)
durability
혹은, contraint를 주는데, 이를 통해 = 0 혹은  ≠0을 정의 (equality, inequality)
svm 에서 margin안은 equality그리고 그 밖은 inequality를 의미함. 
(ref. https://scikit-learn.org/stable/modules/svm.html)

KKT conditions (https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)
Single constraint

이중성 정리 (durability theory) (https://ichi.pro/ko/ijungseong-jeongliwa-geu-jeungmyeong-97308844102052)
min, max가 다 있는것 이는 
convex, concave 와 연관

여기서, convex,concave와 saddle p't사이의 연관성에 대해서 생각해 볼 것
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.178.3064&rep=rep1&type=pdf

-> 연관된것이 SVM의 
generated optimal control 제어 
convex, concave and saddle p't


110321

V(D,G) = E_(x~Pdata(x))[logD(x)]+E(z~Pz(z))[log(1-D(G(z))]
V(D,G) = E_(z~Pz(z))[log[D(x) * (1-D(G(z))]]
여기서 D(x)* (1-D(G(z)))가 의미하는 바는?
D(x)는 확률 
x는 변수, 주사위 던져지는 수를 의미함. 즉, 1 ~ 6
Joint probability -> 베이지안 강의 확인하기

minmax{V(D,G)} => durability 
??? =>  optimization?
D(x)* (1-D(G(z))에서 둘다 만족해야하는 조건이므로, 그냥 convex 혹은 concave로는 도저희 문제를 풀수가 없고, saddle p't처럼 둘다 조건을 만족해야만 하는 조건에서 문제를 풀수 있다. => saddle p't convex, concave (ref. https://www.offconvex.org/2016/03/22/saddlepoints/)

\\ unfortunately, this idea was totally wrong :o( 
saddle point is related with single value decomposition...


cross entropy => https://medium.com/unpackai/cross-entropy-loss-in-ml-d9f22fc11fe0



https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network


https://everyday-deeplearning.tistory.com/entry/%EC%B4%88-%EA%B0%84%EB%8B%A8-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0Generative-Models-GAN
https://jaaamj.tistory.com/85
https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/

=========================

GAN에 대한 설명 : http://jaejunyoo.blogspot.com/2017/07/f-gan-3.html
Why are GANs hard to train? https://www.cs.umd.edu/~tomg/projects/stable_gans/
카카오스토리 GAN : https://brunch.co.kr/@kakao-it/145
지능로봇/인공지능 식 : https://greedywyatt.tistory.com/category/%EC%A7%80%EB%8A%A5%EB%A1%9C%EB%B4%87/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5?page=3
*** Papers ***
Local Saddle Point optimization 
Trainning GAN via primal-dual subgradient methods : a lagrangian perspective on GAN
Fisher GAN

=========================
베이즈정리, ML과 MAP : https://darkpgmr.tistory.com/62
Understand Bayes Rule, Likelihood, Prior and Posterior : https://towardsdatascience.com/understand-bayes-rule-likelihood-prior-and-posterior-34eae0f378c5

110821

probability vs statistics 
probability는 common sense에 기반하고 있고,
statistics는 experience에 기반하고 있다. 
왜 Bayesian theorem이 나왓지?

확률론과 통계학에서, 베이즈 정리(영어: Bayes’ theorem)는 두 확률 변수의 사전 확률과 사후 확률 사이의 관계를 나타내는 정리다. 베이즈 확률론 해석에 따르면 베이즈 정리는 사전확률로부터 사후확률을 구할 수 있다.[1]

베이즈 정리는 불확실성 하에서 의사결정문제를 수학적으로 다룰 때 중요하게 이용된다. 특히, 정보와 같이 눈에 보이지 않는 무형자산이 지닌 가치를 계산할 때 유용하게 사용된다. 전통적인 확률이 연역적 추론에 기반을 두고 있다면 베이즈 정리는 확률임에도 귀납적, 경험적인 추론을 사용한다.[2]
(ref. https://ko.wikipedia.org/wiki/%EB%B2%A0%EC%9D%B4%EC%A6%88_%EC%A0%95%EB%A6%AC)

베이지안이 나온 배경 
1) Frequentist probability (빈도확률)은 시행횟수를 빈도수로 측정하여 나오는 것. 하지만, 화산 폭발과 같이 빈도 확률로 계산하기 힘든 경우. 세상에서 반복할수 없는 사건 즉 '빈도확률'의 개념을 적용할수 없는 사건. 일어나지 않은 일에 대한 확율을 불확실성 (uncertainty)의 개념. 즉, 사건과 관련이 있는 확률을 이용해서 새롭게 일어날 사건을 추정하는 것. 
2) 베이즈정리는 종속적 (의존적)관계에 놓은 사건들을 기반으로 확률을 구함
3) 베이지안 확률은 조건부 확률로 나타내며, 정보를 업데이트하면서 사후확률을 구함. 


ref. https://bioinformaticsandme.tistory.com/47

베이지안 딥러닝에 대한 포괄적 설명 
https://ichi.pro/ko/beijian-dib-leoning-e-daehan-pogwaljeog-in-sogae-19936594383326

베이지안이론 : https://bioinformaticsandme.tistory.com/47
eigendecomposition : https://jeongchul.tistory.com/603
고유벡터, 고윳값 : https://ko.wikipedia.org/wiki/%EA%B3%A0%EC%9C%B3%EA%B0%92%EA%B3%BC_%EA%B3%A0%EC%9C%A0_%EB%B2%A1%ED%84%B0
http://matrix.skku.ac.kr/math4ai/Math4AI.pdf
SVD + saddle p't : https://m.blog.naver.com/skkong89/221828535184
https://angeloyeo.github.io/2020/06/17/Hessian.html
svd : https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm


GAN에서 D(x)가 likelihood인 이유? 


likelihood : https://arxiv.org/pdf/1707.07530.pdf
GAN+ likelihood : https://tobigs.gitbook.io/tobigs/deep-learning/computer-vision/gan-generative-adversarial-network
https://kakalabblog.wordpress.com/2017/07/27/gan-tutorial-2016/


SSV(Singularity, Saddle p't, Value fn)

regression : Lasso, Ridge
Prior probablity

Bayesian theorem : prior (assumption, 어떤 조건) * likelihood = posterior (구하고자하는 바)

