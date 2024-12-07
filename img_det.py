import os
from matplotlib import pyplot as plt
import numpy as np
# import cv2
# import socket
# import json
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os


def imag_pro(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    plt.imshow(img)
    plt.show()
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    print("image path is ", img_path)
    return img_preprocessed


def detect_shapes(img_path):
    directory = os.walk(img_path)
    files = []
    for root, dir, fil in directory:
        for file in fil:
            files.append(os.path.join(root, file))
    model1 = keras.models.load_model(
        'F:\Actual Projects\Capstone\model\CNN_aug_best_weights.h5')
    classNames = ['circle', 'pentagon', 'square', 'star', 'triangle']
    test_preprocessed_images = np.vstack([imag_pro(fn) for fn in files])
    prediction = model1.predict(test_preprocessed_images)

    arr = np.argmax(prediction, axis=1)
    # print("arr is ", arr)
    final = []
    for i in arr:
        final.append(classNames[i])
    print(final)
