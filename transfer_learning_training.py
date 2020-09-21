from torch.autograd import Variable
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import distance
import numpy as np

level_of_blur = "1"

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

#LFW trainig file size is 2200 
#TODO Make sure to use a non remainder batch-size



#Read in lfw data pairs
train_pairs = []
with open("data/pairsDevTrainWithBlur.txt", 'r') as fi:  
    for line in fi:
        if len(line.split()) >= 3:
            train_pairs.append(" " .join(line.split()))


#Prepare Transforms
transform = transforms.Compose(
    [transforms.Resize((224,224)),  # resized to the network's required input size
     transforms.ToTensor()])
     #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Based on ImageNet Mean & STD



#Alternative to torch's built-in dataloader, load the data and train the network
resnet = InceptionResnetV1(pretrained="casia-webface", num_classes=5749)
resnet.load_state_dict(torch.load('trained_models/lvl0/demo500.pth'))
criterion = ContrastiveLoss()
optimizer = optim.Adam(resnet.parameters(),lr = 0.0005)


for epoch in range(1002): 
    batch_size = 64
    #print("Running Epoch " + str(epoch))
    
    for batch_init in range(0,len(train_pairs),batch_size):
        print("Processing number " + str(batch_init) + " out of 2200")

        batch_end = batch_init + batch_size

        input_pair_a = []
        input_pair_b = []

        labels = []
        outputs = []

        #For each Batch, Load images and apply transformations
        for file_name in train_pairs[batch_init:batch_end]:
            file_pieces = file_name.split()

            if len(file_pieces) == 5: # Matching Pair 
                labels.append(0) #0 is true 

                

                pair_a_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
                im_a = Image.open(pair_a_file_name)
                tensor_a  =  transform(im_a)

                pair_b_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[3])) +file_pieces[3]+".jpg"  )
                im_b = Image.open(pair_b_file_name)
                tensor_b  =  transform(im_b)

                input_pair_a.append(tensor_a)
                input_pair_b.append(tensor_b)




            elif len(file_pieces) == 6: # Non-Matching Pair 
                labels.append(1) #1 is false 


                pair_a_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
                im_a = Image.open(pair_a_file_name)
                tensor_a  =  transform(im_a)

                pair_b_file_name = ("data/v2_blurred_lfw/lfw_blur_"+ level_of_blur +"/"+file_pieces[3]+"/"+file_pieces[3]+ "_"+  "0"*(4 - len(file_pieces[4])) +file_pieces[4]+".jpg"  )
                im_b = Image.open(pair_b_file_name)
                tensor_b  =  transform(im_b)


                input_pair_a.append(tensor_a)
                input_pair_b.append(tensor_b)


        labels = np.array(labels)
        labels = torch.from_numpy(labels).float() 
        labels = Variable(labels, requires_grad = True)
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        features_pair_a = resnet(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
        features_pair_b = resnet(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

        loss_contrastive = criterion(features_pair_a,features_pair_b,labels)
        loss_contrastive.backward()
        optimizer.step()
    
        
    if epoch == 0 or epoch % 100 == 0: 
        print("Running Epoch " + str(epoch))

        torch.save(resnet.state_dict(), 'trained_models/500_starting_transfer_0_levels_to_1_level/demo'+ str(epoch) +'.pth')
