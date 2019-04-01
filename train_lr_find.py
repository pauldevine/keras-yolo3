"""
Run LRFinder to find best lr range for our YOLOv3 model.

The result would be saved as 'lr_finder_loss.png'.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from sgd_accum import SGDAccum
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from lr_finder import LRFinder


TOTAL_ITERATIONS = 10000


def _main():
    annotation_path = 'train.txt'
    classes_path = 'model_data/openimgs_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    print('num_classes: ' + str(num_classes))
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    # use darknet53 weights
    #model = create_model(input_shape, anchors, num_classes,
    #        freeze_body=2, weights_path='model_data/darknet53_weights.h5')
    model = create_model(input_shape, anchors, num_classes,
            freeze_body=0, weights_path='logs/002/trained_weights_final.h5')

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    #np.random.seed(10101)
    np.random.shuffle(lines)
    #np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_val = 10000 if num_val > 10000 else num_val
    num_train = len(lines) - num_val

    if True:
        batch_size = 6
        lr_finder = LRFinder(min_lr=1e-10,
                             max_lr=2e-2,
                             steps_per_epoch=TOTAL_ITERATIONS,
                             epochs=1)
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=SGD(lr=1e-8), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('saving trained model')
        model.save('/home/pdevine/kerasfullmodel.h5',overwrite=True,include_optimizer=True)
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, "/home/pdevine/", "my_model.pb", as_text=False)
        print("Saved model to disk")
        newmodel = model.load('/home/pdevine/kerasfullmodel.h5')
        coreml_model = coremltools.converters.keras.convert(newmodel)
        coreml_model.save('my_model.mlmodel')
        print('train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=TOTAL_ITERATIONS,
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=1,
                epochs=1,
                initial_epoch=0,
                callbacks=[lr_finder])

        lr_finder.save_history('lr_finder_loss.csv')
        lr_finder.plot_loss('lr_finder_loss.png')

    # further training if needed.


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

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


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        print(box_data, input_shape, anchors, num_classes)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
