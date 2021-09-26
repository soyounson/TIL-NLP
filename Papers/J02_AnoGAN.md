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

- Generator가 하는 일은 training 데이터와 유사한 '가짜' 이미지를 만들어내는 것입니다. Discriminator가 하는 일은 이미지를 본 뒤, 실제 training 데이터인지 generator로부터 생성된 가짜 이미지인지를 출력하는 것입니다. 트레이닝 동안, generator는 끊임없이 더 나은 가짜 이미지를 생성해 disciminator를 능가하려 노력하는 반면, discriminator는 진짜 이미지와 가짜 이미지를 더 잘 감지하고 분류하기 위해 노력합니다. 이 게임의 균형은 generator가 training 데이터에서 꺼내온 듯한 완벽한 가짜 이미지를 만들어 내고 있을 때이고, discriminator는 항상 generator의 결과가 진짜인지 가짜인지 50%의 신뢰도를 가지고 추측하도록 둔다. Discriminator를 시작으로 튜토리얼을 시작할 텐데, 몇 가지 표기법을 정의해봅시다. 지금부터 'x'를 이미지를 대표하는 데이터라고 부릅시다. D(x)는 x가 generator가 아닌 training 데이터에서 나왔을 확률을 출력하는 discriminator 네트워크이다. 우리는 이미지 데이터를 다루기 때문에 D(x)에 대한 입력의 크기(size)는 3 * 64 * 64입니다. 직관적으로 D(x)는 x가 training 데이터에서 나왔을 때 높고, generator로부터 만들어졌을 때 낮다. D(x)는 전통적인 이진 분류기 (binary classifier)로 생각할 수 있다. Generator를 위한 표기법으로는,  표준정규분포로부터 추출된 잠재 공간(latent space) 벡터를 'z'라고 하자. G(z)는 잠재 벡터 z를 데이터 공간(data-space)으로 매핑시켜주는 generator 함수이다. G의 목표는 training data의 p_data 분포를 추정하여, 그 추정 분포(p_g)로부터 가짜 샘플을 생성하는 것이다. 따라서, D(G(z))는 Generator의 아웃풋이 진짜 이미지일 확률을 의미한다. (* =가짜 이미지가 진짜 이미지로 판별될 확률) Goodfellow의 논문에 묘사된 것처럼,  D와 G는 D는 진짜와 가짜를 정확하게 구분하는 확률(log D(x))을 키우기 위해, G는 D가 G의 아웃풋을 가짜로 판단할 확률(log(1-D(G(x)))을 낮추는 minmax 게임을 한다. (ref. https://comlini8-8.tistory.com/7 )


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
:warning: Instead of a single cost function optimization, it aims at the Nash equilibrium of costs, increasing the representattive power and specificity of the generative model, while at the same time becoming more accurate in classifying real-from generated data and improving the corresponding feature mapping.

#### 2.1 Unsupervised manifold learning of normal anatomical variability
- setup : M개가 세트인 의학 이미지 <I<sub>m</sub>> (healthy anatomy, m = 1...M, size = a x b), 여기서 K 2D image patches <x<sub>k,m</sub>> (size = c x c) 뽑아냄.
- Train : GAN이용해서 manifold X 학습 (healthy anatomy인 <I<sub>m</sub>> 사용)
- Test : <y<sub>n</sub>,l<sub>n</sub>> 여기서 y<sub>n</sub> (size = c x c)은 unseen image이고, l<sub>n</sub>은 an array of binary image-wise ground truth labels, 즉 0 혹은 1이라는 값을 갖은 array임. 
- 즉, test의 경우는 label존재
**Encoding anatomical variability with a GANS** 
: GAN의 구성 및 D/G에 대한 설명
- GAN : two adversarial modules, a generator G + a discriminator D
- G : learn a distribution p<sub>g</sub> over data x via a mapping G(z) of sample z, 1D vectors of uniformly distributed input noise sampled from latent space, Z, to 2D images in the image space manifold X, which is populated by healthy examples.
- G's architecture : convolutional decoder 와 같음.
- D : a standard CNN that maps a 2D image to a single scalar value D(.)
- D/G are simultaneously optimized through the follwoing two-player minimax game with value function V(G,D)

#### 2.2 Mapping new Images to the Latent space
- G는 mapping을 배움 : G(z) = z → x (latent space representations, z → realistic (normal) image, x)
- the degeree of similarity of x and G(z) depends on to which extent the query image following the data distribution p<sub>g</sub> that was used for training of the generator
- define a loss function fot the mapping of new images to the latent space that comprises two components, a **residual loss** and a **discrimination loss**. 
  * **residual loss** enforces the visual similarity between the generated image G(Z<sub>γ</sub>) and query image x
  * **discrimination loss** enforces the generated image G(Z<sub>γ</sub>) to lie on the learned manifold X
- D and G are both utilized to adapt the coefficients of z via backpropagation. (역전파통해 z 계수 적응시키는데 사용됨.)

**Residual loss**
L<sub>R</sub>(Z<sub>γ</sub>) = Σ |x-G(Z<sub>γ</sub>)|

**An improved discrimination loss based on feature matching** (기존 논문과의 차별성)
- 기존에 D를 속이기 위해 Z<sub>γ</sub>를 업데이트 시켰는데, 이 논문에서는 G(Z<sub>γ</sub>)와 매치시키기위해서 Z<sub>γ</sub>를 업데이트 시킴.  이 방법은 feature matching technique에서 영감을 얻었는데, 이는 D부분에서의 overtraining으로 발생하는 GAN의 instability을 다룸.  기존에 D의 output을 극대화시켜 G의 parameters를 옵티마이징시켰던 것에 반해, 여기서는 **G가 training data와 가장 비슷한 statistics를 갖는 데이터를 생성해내도록 force함**. 즉, G부분에 타겟이 맞춰지게 됨. 
- we do not adapt the training objective of the generator during adversarial training, but instead use the idea of feature matching to improve the mapping to the latent space. 
- Instead of using the scalar output of the discriminator for computing the discrimination loss, we propose to use a richer intermediate feature representation of the discriminator and define the discrimination loss:
L<sub>D</sub>(Z<sub>γ</sub>) = Σ |f(x)-f(G(Z<sub>γ</sub>))|

> :warning: the output of an intermediate layer f(.) of the discriminator is used to specify the statistics of an input image. Based on this new loss term, the adaptation of the coordinates of z does not only rely on a hard decision of the trained discriminator, whether or not a generated image G(z<sub>γ</sub>) fits the learned distribution of normal images, but instead takes the rich information of the feature representation, which is learned by the discriminator during adversarial trainning, into account. In this sense our approach utilizes the trained discriminator not as classifier but as a feature extractor. 


- latent space에 맵핑하기 위해, overall loss 는 (weighted sum에 의해 정의)
L(Z<sub>γ</sub>) = (1-λ)L<sub>R</sub>(Z<sub>γ</sub>) + λL<sub>D</sub>(Z<sub>γ</sub>)
여기서 G와 D의 trained paramters는 fixed되어 있고, z만 backpropagation에 의해서 adapt됨.

#### 2.3 Detection of anomalies
여기서는 anomaly score, A(x)이용해서 이상을 발견/탐지함. 
- **anomaly score, A(x)** : express the fit of a query image x to the model of normal images, can be directly derived from the mapping loss function.
A(x) = (1-λ)R(x) + λD(x)
where, R(x) : residual score, residual loss가 마지막 (Γ<sup>th</sup>)에서의 L<sub>R</sub>(Z<sub>Γ</sub>)
       D(x) : discrimination score, discrimination loss가 마지막 (Γ<sup>th</sup>)에서의 L<sub>D</sub>(Z<sub>Γ</sub>)
- anomaly score에 따라서 normal 과 anomal를 구분하는데, 
  large anomaly score : anomalous images 
  small anomaly score : training 과 굉장히 유사한 images, 즉 normal images의미
- **residual image, X<sub>R</sub> = |x-G(Z<sub>Γ</sub>)|** : identification of anomalous regions within an image (이미지 안에서 이상부분을 찾는 것) 
- 추가적으로, **reference anomaly score, A'(x) = (1-λ)R(x) + λD'(x)** where, reference discrimination score D'(x) = L<sub>D'</sub>(Z<sub>γ</sub>)

### :seedling: Chap.3 Experiments
(Data, Data selection and preprocessing, Evaluation, Implementation details, )

### :seedling: Chap.4 Conclusion 
- enable the identification of anomalies on unseen data based on **unsupervised training of a model on healthy data**.
- be able to detect **different known anomalies (retinal fulid, HRF)**.
- be expected to be capable to discover **novel anomalies**
- (discovering anomalies at scale) enables the mining of data for marker candidates subject to further verification. 


