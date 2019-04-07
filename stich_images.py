import cv2
import numpy as np
import os

TEST_DIR='/media/bernal-tensor/out_VTS_03_01_images'
img=[]
os.chdir(TEST_DIR) 
jpgs = [x for x in os.listdir(TEST_DIR) if x.endswith('.jpg')]
print('jpgs[1]:i{}'.format(jpgs[0]))
for jpg in jpgs:
    img.append(cv2.imread(jpg))

height,width,layers=img[1].shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer=cv2.VideoWriter('test.mp4',fourcc,30.0,(720,480))

for image in img:
    writer.write(image)

cv2.destroyAllWindows()
writer.release()
