## :black_heart: ML/DL Methodolgty

explore methodology

### ☺︎ LSTM (Long Short-Term Memory) / CuDNNLSTM
장기 의존성을 학습할 수 있는 RNN(Recurrent Neural Network)의 종류로, RNN과의 차이점은 오랫동안 정보를 기억하는 기능을 갖고 있다. 
기존 Vanilla RNN을 개선한 모델로 데이터의 long-term dependency를 학습하는데 효과적인 모델. 
Gradient explosion, vanishing 문제를 기존 RNN에 비해 줄임으로써 개선된 방법.
ref: https://3months.tistory.com/168

#### LSTM
* datasets scale : 데이터 scale맞추면, weight의 scale도 일관성있게 맞춰짐. MinMaxscale의 경우, 0 =< variables =< 1. 
(ref. https://3months.tistory.com/167)
* Data structure : keras에서 RNN 계열의 모델을 트레이닐 할 때 요구되는 데이터 형식이 존재. (size, timestep, feature)의 3차원 형태의 데이터 형식이어야 함. 
  일반적인 MLP모델의 경우는 size와 feature만 존재하는 2차원 형태지만, RNN의 경우 '시간'이라는 개념이 존재하므로, 한 차원이 늘어나야 함. 
* input shape = (timestep, feature)
* model architecture


#### CuDNNLSTM
* CuDNNLSTM is faster than LSTM (speed up model.evaluate() and model.predict()).
* CuDNNLSTM uses the GPU support. 
* CuDNNLSTM has less options than LSTM.
