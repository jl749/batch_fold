import numpy as np
import tensorflow as tf
from keras.layers import Dense, BatchNormalization
import _PARMs
import json


np.random.seed(_PARMs.NP_SEED)
tf.random.set_seed(_PARMs.TF_SEED)


def logical_xor(a: int, b: int):
    return bool(a) != bool(b)


x = np.random.randint(2, size=[1000, _PARMs.MBATCH_SIZE, 2])
y = np.array([[logical_xor(batch[0], batch[1]) for batch in arr] for arr in x])

model = tf.keras.Sequential([
    Dense(units=2, activation='relu', input_shape=(_PARMs.MBATCH_SIZE, 2)),
    Dense(units=7, activation='relu'),
    BatchNormalization(),
    Dense(units=5, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse')

history = model.fit(x, y, epochs=_PARMs.EPOCHS)

print('input: ', '[[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]')
print('predicted_y: \n', model.predict([[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]))

# save model and smaple test dataset
model.save('example/xor_model')
with open('example/XOR_test_data.txt', 'w') as fs:
    fs.write(json.dumps([[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]))
