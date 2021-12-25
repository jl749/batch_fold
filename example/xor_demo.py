import os

# os.chdir("..")
os.system("python xor_model_generator.py")
os.system("python batch_fold.py -s 5 2 -i example/xor_model -t example/XOR_test_data.txt")
