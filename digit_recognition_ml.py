import os
import copy
import h5py
from glob import glob
from random import shuffle
# https://github.com/bguisard/Digit-Recognition/blob/master/digit_recognition.ipynb
import cv2
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, BatchNormalization, Convolution2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, concatenate, UpSampling2D, Conv2DTranspose

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot

from keras.datasets import mnist

from load import *

img_dir = "C:\\src\\us\\sample_r48_fin\\"
shape = (30, 170, 1)


def convert_label(label):
    n_label = ""
    for digit in label:
        n_digit = str(np.argmax(digit))
        n_label += n_digit
    return n_label


def print_samples():
    # Create samples to test the generator
    i = 0
    for a, b in generator(img_dir, batch_size=16):
        test_imgs = a
        test_lbls = b

        i += 1

        if i > 1:
            break
    rows_to_plot = 8
    cols_to_plot = 2

    f = plt.figure(figsize=(12, 12))

    for i in range(16):
        f.add_subplot(rows_to_plot, cols_to_plot, i+1)
        plt.title(convert_label(np.vstack((test_lbls[0][i], test_lbls[1][i], test_lbls[2][i], test_lbls[3][i], test_lbls[4][i], test_lbls[5][i], test_lbls[6][i], test_lbls[7][i]))))
        plt.imshow(test_imgs[i][:, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_model(input_shape, p=0.5, n_class=10):
    inputs = Input((input_shape[0], input_shape[1], input_shape[2]))

    x = BatchNormalization()(inputs)
    x = Convolution2D(48, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p / 4)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(64, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p / 4)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(128, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p / 2)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(160, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p / 2)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p)(x)

    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Dropout(p)(x)

    # I had to remove this part because the input size we have is too small for a network this deep.
    # Another alternative would have been change the maxpool strides.

    x = BatchNormalization()(x)
    x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(p)(x)

    #x = BatchNormalization()(x)
    #x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    #x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    #x = Dropout(p)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)  # I also reduced the number of activations
    x = Dense(1024, activation='relu')(x)

    c1 = Dense(n_class, activation='softmax')(x)
    c2 = Dense(n_class, activation='softmax')(x)
    c3 = Dense(n_class, activation='softmax')(x)
    c4 = Dense(n_class, activation='softmax')(x)
    c5 = Dense(n_class, activation='softmax')(x)
    c6 = Dense(n_class, activation='softmax')(x)
    c7 = Dense(n_class, activation='softmax')(x)
    c8 = Dense(n_class, activation='softmax')(x)

    output = [c1, c2, c3, c4, c5, c6, c7, c8]

    model = Model(inputs=inputs, outputs=output)

    return model


def convert_output(model_output):
    model_output = np.array(model_output).swapaxes(0, 1)
    labels = []
    for output in model_output:
        label = convert_label(output)
        labels.append(label)

    return labels


def convert_output(model_output):
    model_output = np.array(model_output).swapaxes(0, 1)
    labels = []
    for output in model_output:
        label = convert_label(output)
        labels.append(label)

    return labels


def pred_for_path(path):
    pred_input = np.asarray([
        np.expand_dims(load_img(path)[0], -1)
    ])

    test_pred = model.predict(pred_input)
    return convert_output(test_pred)

# -------------------
model = get_model(input_shape=shape)
model.summary()
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# -------------------
trn_generator = generator(img_dir, batch_size=128)
val_generator = generator(img_dir, batch_size=128)

# -------------------
model.fit_generator(trn_generator,
                    epochs=4,
                    steps_per_epoch=780,
                    validation_data=val_generator,
                    validation_steps=780,
                    verbose=1)


def check_all():
    for f in glob.glob(img_dir + "*.png"):
        truth = ntpath.basename(f).split(".")[0]
        pred = pred_for_path(f)[0]
        if pred != truth:
            print("Invalid match truth:%s pred:%s" %(truth, pred))
