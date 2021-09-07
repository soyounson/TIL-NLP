
## DeepAnT : A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series 
IEEE (2018)
### What is the DeepAnT?
:battery: In contrast to the anomaly detection methods where anomalies are learned, DeepAnT uses unlabled data to capture and learn the data distribution that is used to forecast the normal behavior of a time series. 

→ 일반적인 anomaly detection에서는 이상징후를 타겟으로 하고 있지만, DeepAnT의 경우 Unsupervised learning 기반으로 Unlabeled data에서 정상들의 시간적 흐름을 타겟으로 함. 

:battery: Unsupervised learning 

→ Unsupervised 이므로, anomaly label에 의존하지 않는다. 그래서 실질적으로 라벨링하기가 불가능에 가까운 너무 큰 데이터 셋의 경우에도 무리없이 수행 가능하다.  

:battery: be trained even w/o removing the anomalies from the given data set.

:battery: Two modules of DeepAnT
  - :seedling: time series predictor : use deep convolutional neural network (CNN) to predict the next time stamp on the defined horizon.  (take a time window and predict the next time stamp)
  - :seedling: anomaly detector : be responsible for tagging the corresponding time stamp as normal or abnormal. 

:battery: In most of the cases, the collected data are steaming time series data and due to their intrinsic characteristiscs of periodicity, trend, seasonality, and irregularity, it is a challenging problem to detect point anomalies precisely in them. Furthermore, in most of real life scenarios, it is practically impossible to label enormous amount of data, therefore, we are using an unsupervised method.
→ 대략, 데이터들은 시간의 추이에 따라 기록되어 있고, 이들은 내부에 periodicity, trend, seasonality등의 패턴을 포한하고 있어서, 실제 기존의 anomaly detection이용하여 반복되는 사이클 속에서 point anomalies를 찾기는 불가능하다. 본 연구에서 제안한 비지도학습 베이스로 한 방법은 이러한 모든 것을 포함해서 anomaly detection이 가능하다. 

:battery: The proposed unsupervised approach incorporates context, seasonality, and trend into account for detecting anomalies. 

:battery: Doesn't rely on labeling of anomalies rather it leverages the original time series data even w/o removing anomalies (given that the number of anomalies in the data set is less than 5%).

:battery: Instead of classifying whole time series as normal or abnormal, DeepAnT's objective is to robustly detect point anomalies. In particular, following are the main contributions of this paper: 
- To the best of our knowledge, DeepAnT is the first deep learning based approach which is capable of detecting point anomalies, contextual anomalies, and discords in time series data in an unsupervised setting.
- The proposed pipeline is flexible and can be easily adapted for different use cases and domains. It can be applied to uni-variant as well as multi-variant time series. 
- In contrast to the LSTM based approach, CNN based DeepAnT is not data hungry. It is equally applicable to big data as weel as small data. We are only using 40% of a given time series to train a model.

:battery: 정리하자면, 
* Unsupervised learning -> labeling이 따로 필요없어서 Big data를 다루는데 용이함. 
* time series predictor를 통해, 데이터의 시간적 흐름을 읽으므로 periodicity, trend, seasonality안에 존재하는 anomalies을 찾아내고 detect하는데 용이함. 
* 정상의 흐름을 배우고, 거기서 anomalities 찾는것이므로 anomalies' labels에 너무 의존적이지 않다. 
