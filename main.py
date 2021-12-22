import tensorflow as tf
from keras.layers import Dense, BatchNormalization
import _PARMs
import numpy as np

model = tf.keras.models.load_model('xor_model')

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 5, 2)              6         
_________________________________________________________________
dense_1 (Dense)              (None, 5, 7)              21        
_________________________________________________________________
batch_normalization (BatchNo (None, 5, 7)              28        
_________________________________________________________________
dense_2 (Dense)              (None, 5, 5)              40        
_________________________________________________________________
dense_3 (Dense)              (None, 5, 1)              6         
=================================================================
Total params: 101
Trainable params: 87
Non-trainable params: 14
"""

print(model.predict([[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]))

inputs = tf.keras.Input(shape=(_PARMs.MBATCH_SIZE, 2))
x = inputs

batchNormInfo = []

count = 0
for layer in model.layers[:-1]:
    if isinstance(layer, BatchNormalization):
        batchNormInfo.append({'index': count, 'weights': layer.get_weights()})
        print('####################')
        print(layer.weights)
        print('####################')
    else:
        x = layer(x)
    count += 1
outputs = model.layers[-1](x)

new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
new_model.summary()
for tmp in batchNormInfo:
    index = tmp['index']
    epsilon = 0.0001
    [gamma, beta, mean, var] = tmp['weights']
    [w, b] = new_model.layers[index].get_weights()

    new_model.layers[index].set_weights([w*gamma/np.sqrt(var+epsilon), beta+(b-mean)*gamma/np.sqrt(var+epsilon)])

print(new_model.predict([[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]))