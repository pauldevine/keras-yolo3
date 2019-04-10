import cv2
import numpy as np
import os

TEST_DIR='/media/bernal-tensor/Video_4_Paul/520_1140_process'
img=[]
os.chdir(TEST_DIR) 
jpgs = [x for x in os.listdir(TEST_DIR) if x.endswith('.jpg')]
jpgs.sort()
print('jpgs[1]:i{}'.format(jpgs[0]))
sample = cv2.imread(jpgs[1])

height,width,layers=sample.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer=cv2.VideoWriter('517_1150.mp4',fourcc,30.0,(width,height))

for jpg in jpgs:
    writer.write(cv2.imread(jpg))

cv2.destroyAllWindows()
writer.release()
