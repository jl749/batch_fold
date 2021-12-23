import numpy as np
import tensorflow as tf
from keras.layers import Dense, BatchNormalization
import _PARMs

np.random.seed(0)
tf.random.set_seed(1)


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

# print('x: ', '[[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]')
# print('y: ', model.predict([[[1, 0], [1, 0], [0, 1], [0, 0], [1, 1]]]))

test_hat = model.predict([_PARMs.xor_test_data]).tolist()[0]
print(list(zip(_PARMs.xor_test_data, test_hat)))
model.save('xor_model')
