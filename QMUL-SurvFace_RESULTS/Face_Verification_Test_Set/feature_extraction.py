#from resnet_models import resnet152
from facenet_pytorch import InceptionResnetV1
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import distance
from scipy.io import savemat
import numpy as np
from  sklearn.metrics import roc_auc_score, roc_curve
import random
import numpy
from matplotlib import pyplot as plt
import sklearn.metrics as metrics

#LFW trainig file size is 2200 
#TODO Make sure to use a non remainder batch-size

#Read in lfw data pairs
test_pairs = []
#with open("v2/01/pairsDevTestWithBlur.txt", 'r') as fi: 
#with open("negative_pairs_names.txt", 'r') as fi:
with open("positive_pairs_names.txt", 'r') as fi:
    for line in fi:
        test_pairs.append(line.split())

#Prepare Transforms
transform = transforms.Compose(
    [transforms.Resize((224,224)),  # resized to the network's required input size
     transforms.ToTensor()])
     #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )]) #Based on ImageNet Mean & STD




resnet_blurred_4 = InceptionResnetV1(pretrained="casia-webface", num_classes=5749).eval()
#resnet_blurred_4.load_state_dict(torch.load('v3/augmentation/500_starting_transfer_0_levels_to_2_levels/demo500.pth'))
resnet_blurred_4.load_state_dict(torch.load('../../BINN/v3/mixed_bag/mixed_0_1_2_3_4/demo500.pth'))


scores_unblurred = []
scores_blurred_4 = []
scores_blurred_2 = []
scores_blurred_1 = []

labels = []


    
for name_pair, curent in zip(test_pairs,range(len(test_pairs))):
    print (str(curent)+ "of " + str(len(test_pairs)))

    label = None

    input_pair_a = []
    input_pair_b = []


    pair_a_file_name = ("verification_images/"+name_pair[0][:-3] + "png")
    im_a = Image.open(pair_a_file_name)
    tensor_a  =  transform(im_a)

    pair_b_file_name = ("verification_images/"+name_pair[1][:-3] + "png")
    im_b = Image.open(pair_b_file_name)
    tensor_b  =  transform(im_b)

    input_pair_a.append(tensor_a)
    input_pair_b.append(tensor_b)
    

    
    features_pair_a = resnet_blurred_4(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
    features_pair_b = resnet_blurred_4(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

    euclidean_distance = F.pairwise_distance(features_pair_a, features_pair_b)
    scores_blurred_4.append(euclidean_distance)


results = {"scores": scores_blurred_4}
savemat("pos_pairs.mat", results)


