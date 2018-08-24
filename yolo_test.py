import os
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


FLAGS = None
TEST_DIR = 'open-images-dataset/kaggle-2018-object-detection/test_challenge_2018'


def detect_img(yolo, img_path):
    try:
        image = Image.open(img_path)
    except:
        sys.exit('Cannot open image file: {}'.format(img_path))
    else:
        r_image = yolo.detect_image(image)
        r_image.show()


def detect_test_imgs(yolo):
    jpgs = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    for jpg in jpgs:
        img_path = os.path.join(TEST_DIR, jpg)
        detect_img(yolo, img_path)
        str_in = input('{}, <ENTER> for next or "q" to quit: '.format(img_path))
        if str_in.lower() == 'q':
            break
    yolo.close_session()


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

    FLAGS = parser.parse_args()

    print("Image detection mode")
    detect_test_imgs(YOLO(**vars(FLAGS)))
