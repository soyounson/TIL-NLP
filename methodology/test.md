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
