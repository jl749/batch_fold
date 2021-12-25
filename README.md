# Batch_Fold_assignment


Code was written in python 3.7 and Tensorflow/Keras 2.6

## Preparation
```
pip install -r requirements.txt
```

## How to use
* Full Affine layer NN (XOR example)
```
cd {your_location}/J_Lee_batchFold

# create an example model
python XOR_model_generator.py

# perform batch folding + compare test results
python batch_fold.py -s 5 2 -i example/xor_model -t example/XOR_test_data.txt
```


* CNN (MNIST example)
```
cd {your_location}/J_Lee_batchFold

# create an example model
python MNIST_model_generator.py

# perform batch folding + compare test results
python batch_fold.py -s 28 28 1 -i example/MNIST_model -t example/MNIST_test_data.txt
```

## Examples
* This will execute the example code
```
cd {your_location}/J_Lee_batchFold
python example/{xor_demo.py or MNIST_demo.py}
```
