import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
import numpy as np
import json
import _PARMs


np.random.seed(_PARMs.NP_SEED)
tf.random.set_seed(_PARMs.TF_SEED)

# load dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X[:1000]
train_y = train_y[:1000]
test_X = test_X[:200]
test_y = test_y[:200]

# normalize
train_X, test_X = train_X / 255.0, test_X / 255.0
# summarize loaded dataset
print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))

# reshape dataset to have a single channel
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

# one hot encode target values
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model = tf.keras.Sequential([
    Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1),
                  padding='same', activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=100, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_X, train_y, epochs=_PARMs.EPOCHS, batch_size=32, validation_data=(test_X, test_y), 
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)])

score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_data = np.random.randint(200, size=5)
print('expected_labels: ', np.argmax(test_y[test_data], axis=1))
test_data = test_X[test_data]
print('output: ', model.predict(test_data))

# save model and smaple test dataset
model.save('example/MNIST_model')
with open('example/MNIST_test_data.txt', 'w') as fs:
    fs.write(json.dumps(test_data.tolist()))