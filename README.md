# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

This is a Keras implementation of YOLOv3 (Tensorflow backend), for tackling Kaggle 2018 [Google AI Open Images - Object Detection Track](https://www.kaggle.com/c/google-ai-open-images-object-detection-track) competition. I forked the code from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) and made necessary modification for it to work for the Open Images dataset.

---

## Set up

1. Download the original pre-trained Darknet weights for YOLOv3, and convert it to Keras format.

   ```shell
   $ wget https://pjreddie.com/media/files/darknet53.conv.74 -O darknet53.weights
   $ python3 convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5
   ```

2. Prepare the Open Images dataset for training. Reference: (a) [the Kaggle competition page](https://www.kaggle.com/c/google-ai-open-images-object-detection-track/data), (b) [Overview of the Open Images Challenge 2018](https://storage.googleapis.com/openimages/web/challenge.html)

   After downloading all necessary files, put the dataset into a directory structure like the following. Note that all jpg images for training (and validation) should be located in the `train/` folder. The annotation files such as `challenge-2018-class-descriptions-500.csv` and `challenge-2018-train-annotations-bbox.csv` should be found in the `kaggle-2018-object-detection` folder. And the test jpg images should be in the `kaggle-2018-object-detection/test_challenge_2018` folder.

   ```shell
   $ tree -d ~/data/open-images-dataset
   /home/user/data/open-images-dataset
   ├── kaggle-2018-object-detection
   │   └── test_challenge_2018
   ├── kaggle-2018-visual-relationships
   ├── test
   ├── train
   └── valid
   ```

   Then make a symbolic link in the project folder to the dataset directory. For example,

   ```shell
   $ cd ~/project/keras-yolo3
   $ ln -s ~/data/open-images-dataset .
   ```

## Training

1. Convert Open Images annotations into YOLOv3 format. Here is how YOLOv3 annotations look like:

   One row for one image;
   Row format: `image_file_path box1 box2 ... boxN`;
   Box format: `x_min,y_min,x_max,y_max,class_id` (no space).
   Here is an example:
   ```
   path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
   path/to/img2.jpg 120,300,250,600,2
   ...
   ```

   ```shell
   $ python3 openimgs_annotation.py
   ```

2. Do stage_1 and stage_2 training, with Adam optimizer. The `darknet53.weights` is used to load pre-trained weights. And the trained model weights would be saved as `logs/001/trained_weights_stage_2.h5`.

   ```shell
   $ python3 train1.py
   ```

3. (Optional) Use Learning Rate Finder (LRFinder) to find optimal learning rate range for the YOLOv3 model with the dataset. The result would be saved as `lr_finder_loss.png`. You can then fine-tune the learning rate settings in `train2.py` based on it.

   ```shell
   $ python3 train_lr_find.py
   ```

4. Do the final stage of training, with SGD and "Triangular3" Cyclic Learning Rate (CLR).

   ```shell
   $ python3 train2.py
   ```

## Testing the trained model

   Execute `yolo_test.py` in 'display' mode. By default, the script would use `logs/002/trained_weights_final.h5` as the trained weight for testing.

   ```shell
   $ python3 yolo_test.py --help
   Using TensorFlow backend.
   usage: yolo_test.py [-h] [--model MODEL] [--score SCORE] [--shuffle]
                       [--display] [--submit]

   optional arguments:
     -h, --help     show this help message and exit
     --model MODEL  path to model weight file, default
                    logs/002/trained_weights_final.h5
     --score SCORE  score (confidence) threshold, default 0.001
     --shuffle      shuffle images for display mode
     --display      display mode, to show inferred images with bounding box
                    overlays
     --submit       submit mode, to generate "output.csv" for Kaggle submission
   ```

   For example,

   ```shell
   $ python3 yolo_test.py --display --score 0.3 --shuffle
   ```

## Generating the file for Kaggle submission

   Execute `yolo_test.py` in 'submit' mode. For example,

   ```shell
   $ python3 yolo_test.py --submit
   ```

   Then use `filter_submit.py` to set a different score (confidence) threshold for submission. Note that Kaggle might have problem processing the submission (csv) file if the file contains too many bounding boxes. We can effectively decrease the number of bounding boxes in the submission file by setting a higher score threshold.

   ```shell
   $ python3 filter_submit.py --score 0.01 submit/output.csv submit/output-filtered.csv
   ```

---

## Additional notes

1. The test environment is
   - Python 3.5.2
   - Keras 2.2.2
   - TensorFlow 1.9.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The tiny_yolo3 and train_bottlenet related code was removed, since I do not use them.
