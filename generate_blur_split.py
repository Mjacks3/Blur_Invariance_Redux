from random import seed
from random import random
import random

seed(1504)

"""
for a in range(10):
    print(random.randint(0,4))
"""


original_data_train = []
original_data_test = [] 

with open("data/pairsDevTrain.txt", 'r') as f1: 
    for line in f1:
        original_data_train.append(line)

with open("data/pairsDevTrainWithBlur.txt", 'w+') as f2: 
    for line in original_data_train:
        if  len(line.split()) == 3:
            outputs = line.split()
            f2.write(outputs[0] + "\t")
            f2.write(outputs[1] + "\t")
            f2.write(str(random.randint(0,4)) + "\t")
            f2.write(outputs[2] + "\t")
            f2.write(str(random.randint(0,4)))
            f2.write("\n")

        elif len(line.split()) == 4:
            outputs = line.split()
            f2.write(outputs[0] + "\t")
            f2.write(outputs[1] + "\t")
            f2.write(str(random.randint(0,4)) + "\t")
            f2.write(outputs[2] + "\t")
            f2.write(outputs[3] + "\t")
            f2.write(str(random.randint(0,4)))
            f2.write("\n")

        else:
            f2.write(line)




with open("data/pairsDevTest.txt", 'r') as f1: 
    for line in f1:
        original_data_test.append(line)

with open("data/pairsDevTestWithBlur.txt", 'w+') as f2: 
    for line in original_data_test:
        if  len(line.split()) == 3:
            outputs = line.split()
            f2.write(outputs[0] + "\t")
            f2.write(outputs[1] + "\t")
            f2.write(str(random.randint(0,4)) + "\t")
            f2.write(outputs[2] + "\t")
            f2.write(str(random.randint(0,4)))
            f2.write("\n")

        elif len(line.split()) == 4:
            outputs = line.split()
            f2.write(outputs[0] + "\t")
            f2.write(outputs[1] + "\t")
            f2.write(str(random.randint(0,4)) + "\t")
            f2.write(outputs[2] + "\t")
            f2.write(outputs[3] + "\t")
            f2.write(str(random.randint(0,4)))
            f2.write("\n")

        else:
            f2.write(line)

