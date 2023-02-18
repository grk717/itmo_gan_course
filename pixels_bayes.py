import cv2
import numpy as np
import numpy_indexed as npi
import os
import random

path_to_avatars = r'C:\Users\grk\git\gan_itmo\practice_1\avatars'
files = [os.path.join(path_to_avatars, i) for i in os.listdir(path_to_avatars)]
# read images into the list
imgs = [cv2.imread(filename) for filename in files]
#n = len(imgs) # number of images
# stack arrays over new axis
stacked = np.stack(imgs, axis=2)
h,w,n,c = stacked.shape
# flatten h and w
flattened = stacked.reshape(-1, n, c)


# I decided to take only unique RGB triplets, so i will get only the colors from the original image
# this if a "fair generation"
# one time preparation
big_structure = [] # [[[unique_pixels in x=0y=0], [probabilities]], [[unique_pixels in x=0y=1], [probabilities]], ...]
for coord in range(h*w):
    unique_pixels, counts = npi.unique(flattened[coord], axis=0, return_count=True) # unique pixels in every position
    big_structure.append([unique_pixels, counts / counts.sum()])


# generation
for i in range(5):
    image_as_list = []
    for coord in range(h*w):
        item = random.choices(big_structure[coord][0], weights=big_structure[coord][1])[0]
        image_as_list.append(item)
    arr = np.array(image_as_list).reshape(h,w,c)
    cv2.imwrite(f"smth{i}.jpg", arr)