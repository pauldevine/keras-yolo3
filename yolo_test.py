import os
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np


FLAGS = None
TEST_DIR = 'open-images-dataset/mare_test'
OUTPUT_CSV = 'submit/output.csv'
OUTPUT_DIR = 'submit'


def detect_img(yolo, img_path):
    try:
        image = Image.open(img_path)
    except:
        sys.exit('Cannot open image file: {}'.format(img_path))
    else:
        r_image = yolo.detect_image(image)
        r_image.show()


def detect_test_imgs(yolo):
    global FLAGS
    jpgs = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    if FLAGS.shuffle:
        from random import shuffle
        shuffle(jpgs)
    for jpg in jpgs:
        img_path = os.path.join(TEST_DIR, jpg)
        detect_img(yolo, img_path)
        str_in = input('{}, <ENTER> for next or "q" to quit: '.format(img_path))
        if str_in.lower() == 'q':
            break
    yolo.close_session()


def infer_img(yolo, img_path):
    try:
        image = Image.open(img_path)
    except:
        #sys.exit('Cannot open image file: {}'.format(img_path))
        print('!!! Cannot open image file: {}'.format(img_path))
        return []
    else:
        return yolo.infer_image(image)


def submit_test_imgs(yolo):
    jpgs = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    os.makedirs(os.path.split(OUTPUT_CSV)[0], exist_ok=True)
    with open(OUTPUT_CSV, 'w') as f:
        f.write('ImageId,PredictionString\n')
        for jpg in jpgs:
            print(jpg)
            img_path = os.path.join(TEST_DIR, jpg)
            boxes = infer_img(yolo, img_path)
            f.write('{},'.format(os.path.splitext(jpg)[0]))
            # 1 record: [label, confidence, x_min, y_min, x_max, y_max]
            box_strings = ['{:s} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(b[0], b[1], b[2], b[3], b[4], b[5]) for b in boxes]
            if box_strings:
                f.write(' '.join(box_strings))
            f.write('\n')
    yolo.close_session()


def train_test_imgs(yolo):
    classes_path = 'model_data/openimgs_classes.txt'
    class_names = get_classes(classes_path)
    codes_path = 'model_data/openimgs_codes.txt'
    class_codes = get_classes(codes_path)
    taxonomy = {}
    print('names: {} \ncodes:{}'.format(class_names, class_codes))
    for indx, name in enumerate(class_names):
        print('indx: {}, name: {}'.format(indx,name))
        taxonomy[class_codes[indx]] = name
    print('taxonomy: {}'.format(taxonomy))
      
    jpgs = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for jpg in jpgs:
        print(jpg)
        img_path = os.path.join(TEST_DIR, jpg)
        base = os.path.splitext(jpg)[0]
        xml_file = base + '.xml'
        boxes = infer_img(yolo, img_path)
        image = Image.open(img_path)

        print('Found {} bounding boxes'.format(len(boxes)))
        with open(OUTPUT_DIR + '/' + xml_file, 'w') as f:
          f.write('''
            <annotation>
              <folder>Videos</folder>
              <filename>{}</filename>
              <size>
                  <width>{}</width>
                  <height>{}</height>
                  <depth>3</depth>
              </size>
              <segmented>0</segmented>'''.format(jpg, image.width, image.height))
          f.write('{},'.format(os.path.splitext(jpg)[0]))
          # 1 record: [label, confidence, x_min, y_min, x_max, y_max]
          box_strings = []
          for box in boxes:
            label = taxonomy[box[0]]
            top = int(box[2] * image.height)     #x_min
            left = int(box[3] * image.width)     #y_min
            bottom =int(box[4] * image.height)   #x_max
            right = int(box[5] * image.width)    #y_max
            #print('label: {} top: {} left: {} bottom: {} right: {}'.format(label, 
            #    top, left, bottom, right))
            box_strings.append('''
            <object>
              <name>{}</name>
              <pose>Unspecified</pose>
              <truncated>0</truncated>
              <occluded>0</occluded>
              <difficult>0</difficult>
              <bndbox>
                  <xmin>{}</xmin>
                  <ymin>{}</ymin>
                  <xmax>{}</xmax>
                  <ymax>{}</ymax>
              </bndbox>
            </object>
            '''.format(label,top,left,bottom,right))
          if box_strings:
              f.write('\n'.join(box_strings))
          f.write('</annotation>\n')
    yolo.close_session()


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--score', type=float,
        help='score (confidence) threshold, default ' + str(YOLO.get_defaults("score"))
    )

    parser.add_argument(
        '--shuffle', default=False, action="store_true",
        help='shuffle images for display mode'
    )

    parser.add_argument(
        '--display', default=False, action="store_true",
        help='display mode, to show inferred images with bounding box overlays'
    )

    parser.add_argument(
        '--submit', default=False, action="store_true",
        help='submit mode, to generate "output.csv" for Kaggle submission'
    )

    parser.add_argument(
        '--train', default=False, action="store_true",
        help='create image.xml files to do additional training via algorithm'
    )

    FLAGS = parser.parse_args()

    if FLAGS.display:
        print("Display mode")
        detect_test_imgs(YOLO(**vars(FLAGS)))
    elif FLAGS.submit:
        print("Submit mode: writing to output.csv")
        submit_test_imgs(YOLO(**vars(FLAGS)))
    elif FLAGS.train:
        print("Training mode: writing output to .xml files")
        train_test_imgs(YOLO(**vars(FLAGS)))
    else:
        print("Please specify either Display, Submit, or train mode.")
