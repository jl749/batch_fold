from typing import List
import tensorflow as tf
from keras.layers import BatchNormalization, InputLayer
import numpy as np
import argparse
import json


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shape", nargs='+', type=int, required=True,
                help="input shape e.g. -s 255, 255")
ap.add_argument("-i", "--input", required=True,
                help="base path to model directory\n"
                     "e.g. -i example/xor_model")
ap.add_argument("-e", "--epsilon", type=float, default=0.0001,
                help="epsilon value to be used for batch folding")
ap.add_argument("-t", "--test", default=None,
                help="test data to compare input model and batch folded model results\n"
                     "e.g. -i example/MNIST_test_data.txt")
args = vars(ap.parse_args())

model = tf.keras.models.load_model(args['input'])
print("===FROM===")
model.summary()

epsilon: float = args['epsilon']
input_shape: List[int] = [num for num in args['shape']]
# input_shape.insert(0, _PARMs.MBATCH_SIZE)

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

if args['test']:
    with open(args['test']) as fs:
        test_data = json.load(fs)
        print('before: \n', np.argmax(model.predict(test_data), axis=1))
        print('after: \n', np.argmax(new_model.predict(test_data), axis=1))
        # print('before: \n', model.predict(test_data))
        # print('after: \n', new_model.predict(test_data))
