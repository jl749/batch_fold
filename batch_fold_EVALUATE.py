import tensorflow as tf
import argparse
import _PARMs

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
                help="model name to be tested"
                     "e.g. -i xor_model")
args = vars(ap.parse_args())

model = tf.keras.models.load_model(args['name'])
new_model = tf.keras.models.load_model(args['name']+'_FOLDED')

print(model.predict(_PARMs.test_data))
print(new_model.predict(_PARMs.test_data))
