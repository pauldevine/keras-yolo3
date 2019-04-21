import coremltools
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'mare_train.txt'
    log1_dir = 'logs/001/'
    log2_dir = 'logs/002/'
    classes_path = 'model_data/openimgs_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    print("CLASS NAMES: " + str(class_names))
    num_classes = len(class_names)
    print("CLASS NAMES: " + str(class_names) + " " + str(num_classes))
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes,
            freeze_body=0, weights_path=log2_dir+'trained_weights_final.h5')

    coreml_model = coremltools.converters.keras.convert(
            model,
            input_names="image",
            image_input_names="image",
            output_names="output",
            add_custom_layers=True,
            custom_conversion_functions={ "Lambda": Lambda })

    # Fill in the metadata and save the model.
    coreml_model.author = "Paul Devine"
    coreml_model.license = "Commercial"
    coreml_model.short_description = "Trying export of fish model"
    coreml_model.input_description["image"] = "Input image"
    coreml_model.output_description["output"] = "The predictions"
    coreml_model.save("/Users/pdevine/projects/keras-yolo3/fish_model.mlmodel")

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

if __name__ == '__main__':
    _main()