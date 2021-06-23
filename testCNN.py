import os
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing
from keras.models import load_model
from glob import glob
import tensorflow as tf
tf.disable_v2_behavior()
tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
# (silence)


#CNN Model will determine if picture is of a drone

def cnn_dronePred(img_test):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    class_names = ['DRONE', 'NO_DRONE']
    width = 96
    height = 96

    model = load_model('drone_detector.h5')

    base_path_img = "static/images"
    images = []
    path = os.path.join(base_path_img, img_test)

    for image_path in glob(path):
        img = preprocessing.image.load_img(image_path, target_size=(width, height))

    img_X = np.expand_dims(img, axis=0)

    predictions = model.predict(img_X)
    result = class_names[np.argmax(predictions)]

    print('The type predicted is: {}'.format(result))

    return result
