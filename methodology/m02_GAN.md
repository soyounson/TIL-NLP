## :black_heart: ML/DL Methodolgty

### ☺︎ GAN (Generative Adversarial Networks)
GANs are comprised of both generator and discriminator models. 
The generator is responsible for generating new samples from the domain, and the discriminator is responsible for classifying whether samples are real or fake (generated). Importantly, the performance of the discriminator model is used to update both the model weights of the discriminator itself and the generator model. This means that the generator never actually sees examples from the domain and is adapted based on how well the discriminator performs.

### 1D GAN

ref: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

1. Select a One-Dimensional Function
2. Define a Discriminator Model
3. Define a Generator Model
4. Training the Generator Model
5. Evaluating the Performance of the GAN
6. Complete Example of Training the GAN
#### Discriminator model 

* This is a binary classification problem. 
  - Input : Sample with two real values 
  - Output : Binary classification, likelihood the same is real (or fake)
* one hidden layer w/ 25 nodes w/ an activation funce of ReLU and weight initialization method of He Weight initiallization
* output layer w/ 1 node for a binary classification using the sigmoid activation function 
* minimize the binary cross entropy loss function and Adam version of stochastic gradient descent 

```
# define the discriminator model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
 
# define the standalone discriminator model
def define_discriminator(n_inputs=2):
	model = Sequential()
  #..... hidden layer w/ 25 nodes
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# define the discriminator model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
```
The model summary is 
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_12 (Dense)             (None, 25)                75        
_________________________________________________________________
dense_13 (Dense)             (None, 1)                 26        
=================================================================
Total params: 101
Trainable params: 101
Non-trainable params: 0
_________________________________________________________________
```

