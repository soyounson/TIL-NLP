
## DeepAnT : A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series 
IEEE (2018)
### Abstract
:battery: In contrast to the anomaly detection methods where anomalies are learned, DeepAnT uses unlabled data to capture and learn the data distribution that is used to forecast the normal behavior of a time series. 

→ 일반적인 anomaly detection에서는 이상징후를 타겟으로 하고 있지만, DeepAnT의 경우 Unsupervised learning 기반으로 Unlabeled data에서 정상들의 시간적 흐름을 타겟으로 함. 

:battery: Unsupervised learning 

→ Unsupervised 이므로, anomaly label에 의존하지 않는다. 그래서 실질적으로 라벨링하기가 불가능에 가까운 너무 큰 데이터 셋의 경우에도 무리없이 수행 가능하다.  

:battery: be trained even w/o removing the anomalies from the given data set.

:battery: Two modules of DeepAnT
  - :seedling: time series predictor : use deep convolutional neural network (CNN) to predict the next time stamp on the defined horizon.  (take a time window and predict the next time stamp)
  - :seedling: anomaly detector : be responsible for tagging the corresponding time stamp as normal or abnormal. 
