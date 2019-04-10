import cv2
import numpy as np
import os

TEST_VIDEO = '/media/bernal-tensor/Video_4_Paul/520_1140.VOB'
TEST_DIR = '/media/bernal-tensor/Video_4_Paul/520_1140_raw/'

import cv2
vidcap = cv2.VideoCapture(TEST_VIDEO)
path, video_name = os.path.split(TEST_VIDEO)
base_name = video_name.split('.')[0]
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(TEST_DIR + "{}_frame_{:06d}.jpg".format(base_name, count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 