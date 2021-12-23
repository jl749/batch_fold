import os

os.chdir("..")
os.system("python xor_model_generator.py")
os.system("python batch_fold.py -s 2 -i xor_model")
