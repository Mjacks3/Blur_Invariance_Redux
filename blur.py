import sys
import cv2
import os

source_dir = "data/cropped_lfw/lfw_blur_"
#source_dir = "data/example_lfw/demo_blur_"

kernel_factor = 6 

for blur_depth in range(1,5):
    sigma = 3 
    kernel = None

    for a in range(blur_depth - 1):
        sigma *= 1.6
        
    kernel = kernel_factor * sigma

    sigma = int(sigma)
    print(sigma)

    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1

    print(kernel)
    
    for r, d, f in os.walk(source_dir+str(blur_depth)):
        for fi in f:
            full_path = r + "/"+ fi
            image = cv2.imread(full_path)

            for iterations in range(blur_depth):
                #Applies Guassian Blur to dataset n times, where N = blur_depth

                image = cv2.GaussianBlur(image,(kernel,kernel),sigma)
                
            cv2.imwrite(full_path,image)
    
