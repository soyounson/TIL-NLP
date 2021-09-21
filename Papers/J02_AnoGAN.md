## Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide marker Discovery (2017)

### :seedling: Background 
#### :leaves: GAN (Generative Adversarial Network)
- 딥러닝 모델 중 이미지 생성에 널리 쓰이는 모델이다. 기본적인 딥러닝 모델 CNN (Convolutional NEural NEtwork)과 비교하면, CNN은 이미지 분류 (image classification)에 쓰이는데 반해 GAN은 데이터셋과 유사한 이미지를 만드는 것을 목적으로 한다. 
- Generator (생성자)와 (판별자), 두개의 모델이 동시에 적대적인 과정으로 학습한다. 즉, 생성자는 실제 데이터 분포를 학습하고, 판별자는 원래의 데이터인지 생성자로부터 생성된 것인지 구분한다. 생성자의 학습과정은 이미지를 잘 생성해서 속일 확률을 높이고, 판별자는 제대로 구분하는 확률을 높이는 두 플레이어의 반복되는 minmax game 과정이다. 
(ref : https://ysbsb.github.io/gan/2020/06/17/GAN-newbie-guide.html)
- D는 전첵적인 식을 크게 만드는 방향으로 파라미터를 학습, D(x)는 크게 만들어야 좋고, D(G(z))는 작게 만들어야 좋다. (ref : https://wingnim.tistory.com/49)
- 생성대립 신경망/생성적 적대 신경망은 미분 가능 생성자에 기초한 또 다른 생성 모형화 접근 방식이다. 
- 생성자망은 표본 x = g(z; θ<sup>(g)</sup>) 들을 직접 생성한다. 반면, 대립자인 판별자 망은 훈련집합에서 뽑은 표본과 생성한 표본을 구분하려 한다. 판별자망은 하나의 확률값 d(x; θ<sup>(d)</sup>)을 산출하는데, 이 확률값은 x가 모형에서 뽑은 가짜 표본이 아니라 진짜 훈련 견본일 확률을 나타낸다. 생성 대립 신경망의 학습 과정은 영합게임 (zero-sum game)의 관점에서 설명하는 것이 가장 쉽다. 영합게임에서 함수 v(θ<sup>(g)</sup>, θ<sup>(d)</sup>)는 판별자가 받는 보상을 결정하고, 생성자는 그것의 부정인 -v(θ<sup>(g)</sup>, θ<sup>(d)</sup>)를 보상으로 받는다. 
- 수렴시 생성자의 표본은 실제 자료와 구분할 수 없을 정도이며, 판별자는 모든 점에서 1/2의 확률을 출력한다. 그러면 판별자를 폐기해도 된다. 
(ref : Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep learning (2015))

#### :leaves: GAN (Generative Adversarial Network) models 
- GAN : 태초의 GAN은 FC layer을 사용. generator는 CNN을 사용해서 만듦.
- DCGAN : pooling layer 사용하지 않고, stride size = 2로 해결함. bath norm, adam 사용 
(ref : https://wingnim.tistory.com/49)

> 가장 먼저 정상 데이터들로 DCGAN릉 학습시키고, anomaly 판단. P_z에서 z를 뽑고, Z의 계수를 업데이트하는 과정을 일정 거친 다음 이 z로부터 query data, x가 다시 만들어지는지를 판단함. 즉, 정상 데이터의 latent space로 적절하게 매핑되는지 여부를 통해 데이터의 정상 여부를 판단. (ref : https://kangbk0120.github.io/articles/2018-01/ano-gan)

### :seedling: Prologue (정리)
- background : 의학쪽 이미지 데이터에서 진단, 질병 진행 모니터링 및 진행 반응등을 확인하기 위해 사용.
- related work : anomaly detection이란 정상적 데이터에 fit되지 않는 것들을 발견해내는 과정임. 
- unsupervised learning을 베이스로 한 DCGAN임.
- 건강한 상태의 이미지를 이용하여 Discriminator를 training시킴.
- 그 다음으로, discriminator를 통해 건강한 데이터와 이상이 있는 데이터에서 이상 진단을 하도록 함. 
- image space에서 latent space로 맵핑 시킴

### :seedling: Abstract
- perform **unsupervised learning to identify anomalies in imaging data as candidates for markers**.
- propose **AnoGAN**, a deep convolutional generative adversarial network (DCGAN) to learn a minifold of normal anatomical variability, accompanying a novel anomaly scoring scheme based on the **mapping from image space to a latent space**. 

### :seedling: Chap.1 Introduction 
#### :leaves: motivation 
- the detection and qualification of disease markers in imaging data is critical during diagnosis, and monitoring of disease progression, or treatment response.
- medical imaging enables the observation of markers correlating with disease status, and treatment response. 
- typically,computational detection in imaging data requires **extensive supervised training using large amounts of annotated data such as labeled lesions**
#### :leaves: related work
- anomaly detection is the task of **identifying test data not fitting the normal data distribution seen during training**.
- typically either use an explicit representation of the **distribution of normal data in a feature space, and determine outliers based on the local density** at the observations' position in the feature space. 
- Seebock et al identified anomalous regions in optical coherence tomography (OCT) images through unsuperviesd learning on healthy examples, using a **convolutional autoencoder and a one-class support vectror machine (SVM)**, and eplored different classes of anomalies. In contrast to this work, the SVM in Seebock's sutdy involved the need to choose a hyper-paramter that defined the amount of training points covered by the estimated healthy region. 
- RAdford et al. introduced deep convolutional generative adversarial networks (DCGANs) and showed that GANs are capacble of captureing semantic image content enabling vector arithmetic for visual concepts.
- Yeh et al. trained GANs on natural images and applied the trained model for semantic image inpainting. 

#### :leaves: distinction of this work
- define an anomaly score, which is not needed in an inpainting task. 
- the main difference of this paper to aforementioned anomaly detection work is the representative power of the generative model and the coupled mapping schema, which utilizes a trained DCGAN and **enable accurate discrimination between normal anatomy, and local anomalous appearance**. (:warning: how?)


### :seedling: Chap.2 Generative Adversarial Representation Learning to Identify Anomalies



### :seedling: Chap.3 Experiments
(Data, Data selection and preprocessing, Evaluation, Implementation details, )

### :seedling: Chap.4 Conclusion 
- enable the identificatio nof anomalies on unseen data based on **unsupervised training of a model on healthy data**.
- be able to detect **different known anomalies (retinal fulid, HRF)**.
- be expected to be capavle to discover **novel anomalies**
- (discovering anomalies at scale) enables the mining of data for marker candidates subject to further verification. 


