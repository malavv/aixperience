import os
import copy
import h5py
import cPickle as pickle
from glob import glob
from random import shuffle

import cv2
from PIL import Image
from IPython.display import SVG

import numpy as np
import pandas as pd

import sklearn.utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def create_numbers(numbers, number_labels, maxlength=5, digit_sz=(28, 28), return_label=False):
    # Attention: Only coded to work with grayscale images at the moment.

    # Randomly choose a number length:
    img_len = 5
    label = np.empty(5, dtype='str')

    # print "length: ", img_len

    # Randomly choose where in our image the sequence of numbers will appear
    if img_len < maxlength:
        st_point = np.random.choice(maxlength - img_len)
    else:
        st_point = 0

    # print "start:", st_point

    charmap = np.zeros(maxlength)
    charmap[st_point:st_point + img_len] = 1

    # print "charmap: ", charmap

    # Define a blank character - this will ensure our input image always have the same dimensions
    blank_char = np.zeros_like(digit_sz)
    blank_lbl = "."

    # Initialize a blank image with maxlen * digit_dz width and digit_sz height
    new_img_len = maxlength * digit_sz[1]
    new_img = np.zeros((digit_sz[0], new_img_len))

    # Fill in the image with random numbers from dataset, starting at st_point
    for i, b in enumerate(charmap):
        if b > 0:
            n = np.random.choice(len(numbers))
            st_pos = i * digit_sz[1]
            new_img[:, st_pos:st_pos + digit_sz[1]] = numbers[n]
            label[i] = str(number_labels[n])
        else:
            label[i] = blank_lbl

    if return_label:
        return new_img, label

    return new_img


def generator(numbers, number_labels, batch_size=32):
    """
    This generator receives mnist digits and labels and returns a batch for training

    Input:
    numbers - array with mnist images.
    number_labels - array with mnist labels.

    Arguments:
    batch_size - size of the mini batch

    Output:
    X_train and y_train
    """
    while True:  # Loop forever so the generator never terminates

        images = []
        labels = []

        for batch_sample in range(batch_size):
            img, label = create_numbers(numbers, number_labels, return_label=True)

            # Here we will convert the label to a format that Keras API can process:
            n_label = np.zeros((5, 11), dtype='int')
            for i, digit in enumerate(label):
                if digit == ".":
                    n_digit = 10
                else:
                    n_digit = int(digit)

                n_label[i][n_digit] = 1

            images.append(img)
            # labels.append(label)
            labels.append(n_label)

        X_train = np.array(images)
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, -1)

        y_temp = np.array(labels)

        y1 = y_temp[:, 0, :]
        y2 = y_temp[:, 1, :]
        y3 = y_temp[:, 2, :]
        y4 = y_temp[:, 3, :]
        y5 = y_temp[:, 4, :]

        yield X_train, [y1, y2, y3, y4, y5]

def convert_label(label):
    n_label = ""
    for digit in label:
        if np.argmax(digit) == 10:
            n_digit = "."
        else:
            n_digit = str(np.argmax(digit))
        n_label += n_digit
    return n_label


def print_samples():
    # Create samples to test the generator
    i = 0
    for a, b in generator(x_train, y_train, batch_size=16):
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
        plt.title(convert_label(np.vstack((test_lbls[0][i], test_lbls[1][i], test_lbls[2][i],
                                                    test_lbls[3][i],test_lbls[4][i]))))
        plt.imshow(test_imgs[i][:, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_model(input_shape=(28, 28 * 5, 3), p=0.5, n_class=11):
    inputs = Input(((input_shape[0], input_shape[1], input_shape[2])))

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

    # x = BatchNormalization()(x)
    # x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # x = Dropout(p)(x)

    # x = BatchNormalization()(x)
    # x = Convolution2D(192, 5, activation='relu', padding='same', strides=(1, 1))(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    # x = Dropout(p)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)  # I also reduced the number of activations
    x = Dense(1024, activation='relu')(x)

    c1 = Dense(n_class, activation='softmax')(x)
    c2 = Dense(n_class, activation='softmax')(x)
    c3 = Dense(n_class, activation='softmax')(x)
    c4 = Dense(n_class, activation='softmax')(x)
    c5 = Dense(n_class, activation='softmax')(x)

    output = [c1, c2, c3, c4, c5]

    model = Model(inputs=inputs, outputs=output)

    return model

def convert_output(model_output):
    model_output = np.array(model_output).swapaxes(0, 1)
    labels = []
    for output in model_output:
        label = convert_label(output)
        labels.append(label)

    return labels


# -------------------
model = get_model(input_shape=(28, 28*5, 1))
model.summary()

# -------------------
model = get_model(input_shape=(28, 28*5, 1))
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# -------------------
trn_generator = generator(x_train, y_train, batch_size=128)
val_generator = generator(x_test, y_test, batch_size=128)

# -------------------
model.fit_generator(trn_generator,
                    epochs=2,
                    steps_per_epoch=780,
                    validation_data=val_generator,
                    validation_steps=780,
                    verbose=1)
