import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import coremltools
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_head
from yolo3.utils import get_random_data


def _main():
  classes_path = 'model_data/openimgs_classes.txt'
  class_names = get_classes(classes_path)
  print("[INFO] loading model...")
  # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  model = load_model('/home/pdevine/kerasfullmodel.h5')
  print("[INFO] converting model")

  # json_file = open('/Users/pdevine/projects/keras-tf-machine/keras-yolo3-v2/model.json', 'r')
  # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  #   loaded_model_json = json_file.read()
  #   json_file.close()
  #   model = model_from_json(loaded_model_json)
  # load weights into new model
  # model.load_weights("/Users/pdevine/projects/keras-tf-machine/keras-yolo3-v2/model.h5")
  print("Loaded model from disk")
  coreml_model = coremltools.converters.keras.convert(model,
    input_names="image",
    image_input_names="image",
    image_scale=1/255.0,
    class_labels=class_names,
      is_bgr=True)


def get_classes(classes_path):
  '''loads the classes'''
  with open(classes_path) as f:
      class_names = f.readlines()
  class_names = [c.strip() for c in class_names]
  return class_names


if __name__ == '__main__':
    _main()
