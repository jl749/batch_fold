import tensorflow as tf
from keras.layers import Dense, BatchNormalization
import _PARMs
import numpy as np

model = tf.keras.models.load_model('xor_model')
# model.summary()

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

# inputs = tf.keras.Input(shape=(2,), batch_size=PARMs.MBATCH_SIZE)
# x = Dense(units=2, activation='relu', input_shape=(PARMs.MBATCH_SIZE, 2))(inputs)
# x = Dense(units=7, activation='relu')(x)
# x = Dense(units=5, activation='relu')(x)
# outputs = Dense(units=1, activation='sigmoid')(x)
# new_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# new_model.layers[i].set_weights(listOfNumpyArrays)    

# new_model.summary()

inputs = tf.keras.Input(shape=(_PARMs.MBATCH_SIZE, 2))
x = inputs

batchNormInfo = []

for i, layer in enumerate(model.layers[:-1]):
    if isinstance(layer, BatchNormalization):
        batchNormInfo.append({'index': i, 'weights': layer.get_weights()})
        print('####################')
        print(layer.weights)
        print('####################')
    else:
        x = layer(x)
        # tmp = layer.weights
        # print(tmp, end='\n\n\n')
outputs = model.layers[-1](x)

new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
new_model.summary()
for tmp in batchNormInfo:
    index = tmp['index']
    scale = tmp['weights'][2]  # batch_normalization/moving_mean
    shift = tmp['weights'][3]  # batch_normalization/moving_variance

    [w, b] = new_model.layers[index].get_weights()
    print(b+shift)
    new_model.layers[index].set_weights([np.matmul(w, np.tile(scale,(7,1))), b+shift])

print(new_model.predict([[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]))