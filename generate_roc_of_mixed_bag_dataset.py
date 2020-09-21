from resnet_models import resnet152
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
import numpy as np
from  sklearn.metrics import roc_auc_score, roc_curve
import random
import numpy
import matplotlib
from matplotlib import pyplot as plt
import sklearn.metrics as metrics


font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

level_of_blur = "0"

#LFW trainig file size is 2200 
#TODO Make sure to use a non remainder batch-size



#Read in lfw data pairs
test_pairs = []
with open("v2/01/pairsDevTestWithBlur.txt", 'r') as fi: 
#with open("data/pairsDevTestWithBlur.txt", 'r') as fi:  
    for line in fi:
        if len(line.split()) >= 3:
            test_pairs.append(" " .join(line.split()))


#Prepare Transforms
transform = transforms.Compose(
    [transforms.Resize((224,224)),  # resized to the network's required input size
     transforms.ToTensor()])
     #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) )]) #Based on ImageNet Mean & STD



#Alternative to torch's built-in dataloader, load the data and train the network
#resnet = resnet152()
#resnet.load_state_dict(torch.load('trained_models/unblurred/demo72.pth'))
#resnet.eval()



resnet_blurred_4 = InceptionResnetV1(pretrained="casia-webface", num_classes=5749).eval()
resnet_blurred_4.load_state_dict(torch.load('v3/mixed_bag/mixed_0_1_2_3_4/demo500.pth'))

resnet_blurred_2 = InceptionResnetV1(pretrained="casia-webface", num_classes=5749).eval()
resnet_blurred_2.load_state_dict(torch.load('v3/mixed_bag/mixed_0_1_2/demo500.pth'))

resnet_blurred_1 = InceptionResnetV1(pretrained="casia-webface", num_classes=5749).eval()
resnet_blurred_1.load_state_dict(torch.load('v3/mixed_bag/mixed_0_1/demo500.pth'))

resnet_unblurred = InceptionResnetV1(pretrained="casia-webface", num_classes=5749).eval()
resnet_unblurred.load_state_dict(torch.load('v3/mixed_bag/unmixed/demo500.pth'))


scores_unblurred = []
scores_blurred_4 = []
scores_blurred_2 = []
scores_blurred_1 = []

labels = []
#For each Batch, Load images and apply transformations

for file_name, curent in zip(test_pairs, range(len(test_pairs))):
    print (str(curent)+ "of " + str(len(test_pairs)))

    label = None

    input_pair_a = []
    input_pair_b = []


    file_pieces = file_name.split()

    if len(file_pieces) == 5: # Matching Pair 
        label = 0 #0 is true 

        pair_a_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
        im_a = Image.open(pair_a_file_name)
        tensor_a  =  transform(im_a)

        pair_b_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[3])) +file_pieces[3]+".jpg"  )
        im_b = Image.open(pair_b_file_name)
        tensor_b  =  transform(im_b)

        input_pair_a.append(tensor_a)
        input_pair_b.append(tensor_b)




    elif len(file_pieces) == 6: # Non-Matching Pair 
        label = 1 #1 is false 
        pair_a_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
        im_a = Image.open(pair_a_file_name)
        tensor_a  =  transform(im_a)

        pair_b_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[3]+"/"+file_pieces[3]+ "_"+  "0"*(4 - len(file_pieces[4])) +file_pieces[4]+".jpg"  )
        im_b = Image.open(pair_b_file_name)
        tensor_b  =  transform(im_b)

        input_pair_a.append(tensor_a)
        input_pair_b.append(tensor_b)



    else: 
        print("Error Processing Line: ")
        print(line) 
        exit()


    features_pair_a = resnet_unblurred(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
    features_pair_b = resnet_unblurred(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

    euclidean_distance = F.pairwise_distance(features_pair_a, features_pair_b)
    scores_unblurred.append(euclidean_distance)


    features_pair_a = resnet_blurred_4(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
    features_pair_b = resnet_blurred_4(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

    euclidean_distance = F.pairwise_distance(features_pair_a, features_pair_b)
    scores_blurred_4.append(euclidean_distance)



    features_pair_a = resnet_blurred_2(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
    features_pair_b = resnet_blurred_2(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

    euclidean_distance = F.pairwise_distance(features_pair_a, features_pair_b)
    scores_blurred_2.append(euclidean_distance)



    features_pair_a = resnet_blurred_1(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
    features_pair_b = resnet_blurred_1(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

    euclidean_distance = F.pairwise_distance(features_pair_a, features_pair_b)
    scores_blurred_1.append(euclidean_distance)


    labels.append(label)


#prediction.append(1)
#labels.append(1)'

#optimal_idx = np.argmax(tpr - fpr)
#optimal_threshold = thresholds[optimal_idx]

#print(optimal_threshold)
fpr_u, tpr_u, thresholds_u = roc_curve(labels, scores_unblurred)
fpr_b_4, tpr_b_4, thresholds_b_4 = roc_curve(labels, scores_blurred_4)
fpr_b_2, tpr_b_2, thresholds_b_2 = roc_curve(labels, scores_blurred_2)
fpr_b_1, tpr_b_1, thresholds_b_1 = roc_curve(labels, scores_blurred_1)

roc_auc_u = metrics.auc(fpr_u, tpr_u)
roc_auc_b_4 = metrics.auc(fpr_b_4, tpr_b_4)
roc_auc_b_2 = metrics.auc(fpr_b_2, tpr_b_2)
roc_auc_b_1 = metrics.auc(fpr_b_1, tpr_b_1)

# method I: plt
plt.title('ROC for LFW View 1 (Test Set is '+level_of_blur+' Level(s) of Blur) ')
plt.plot(fpr_b_4, tpr_b_4, 'b', label = '(Randomized 0-4 Levels of Blur): AUC = %0.2f' % roc_auc_b_4)
plt.plot(fpr_b_2, tpr_b_2, 'g', label = '(Randomized 0-2 Levels of Blur): AUC = %0.2f' % roc_auc_b_2)
plt.plot(fpr_b_1, tpr_b_1, 'm', label = '(Randomized 0-1 Levels of Blur): AUC = %0.2f' % roc_auc_b_1)
plt.plot(fpr_u, tpr_u, 'r', label = 'Unblurred Model: AUC = %0.2f' % roc_auc_u)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

