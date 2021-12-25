import os

# os.chdir("..")
os.system("python MNIST_model_generator.py")
os.system("python batch_fold.py -s 28 28 1 -i example/MNIST_model -t example/MNIST_test_data.txt")