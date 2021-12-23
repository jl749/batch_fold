from typing import List
import tensorflow as tf
from keras.layers import BatchNormalization, InputLayer
import _PARMs
import numpy as np
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shape", nargs='+', type=int, required=True,
                help="input shape without mini_batch size (mini_batch in _PARAMs.py)\n"
                     "e.g. -s 255, 255")
ap.add_argument("-i", "--input", required=True,
                help="base path to model directory\n"
                     "e.g. -i xor_model")
ap.add_argument("-e", "--epsilon", type=float, default=0.0001,
                help="epsilon value to be used for batch folding")
args = vars(ap.parse_args())

model = tf.keras.models.load_model(args['input'])
print("===FROM===")
model.summary()

epsilon: float = args['epsilon']
input_shape: List[int] = [num for num in args['shape']]
input_shape.insert(0, _PARMs.MBATCH_SIZE)

inputs = tf.keras.Input(shape=tuple(input_shape))
x = inputs

batchNormInfo = []

count = 0
for layer in model.layers[:-1]:
    if isinstance(layer, InputLayer):
        count += 1
        continue
    elif isinstance(layer, BatchNormalization):
        batchNormInfo.append({'index': count, 'weights': layer.get_weights()})
        # print('####################')
        # print(layer.weights)
        # print('####################')
    else:
        x = layer(x)
        count += 1
outputs = model.layers[-1](x)

new_model = tf.keras.Model(inputs=inputs, outputs=outputs)

print("\n====TO====")
new_model.summary()
for tmp in batchNormInfo:
    index = tmp['index']
    [gamma, beta, mean, var] = tmp['weights']
    [w, b] = new_model.layers[index].get_weights()

    new_model.layers[index].set_weights([w*gamma/np.sqrt(var+epsilon), beta+(b-mean)*gamma/np.sqrt(var+epsilon)])

new_model.save(args['input']+'_FOLDED')
