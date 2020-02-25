import glob
import ntpath
from random import sample

import cv2
import numpy as np
from keras.utils.np_utils import to_categorical


def load_img(path):
    filename_code = ntpath.basename(path).split('.')[0]
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.bitwise_not(im)
    return im, to_categorical([int(c) for c in filename_code], 10)


def generator(folder, batch_size=32):
    all_files = glob.glob(folder + '*.png')
    while True:  # Loop forever so the generator never terminates
        images = []
        labels = []

        for batch_sample in range(batch_size):
            filename = sample(all_files, 1)[0]
            img, label = load_img(filename)

            images.append(img)
            labels.append(label)

        x_train = np.array(images)
        # making shape (h, w, 1) instead of (h, w), no ideal why needed but break if not.
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)

        y_temp = np.array(labels)

        y1 = y_temp[:, 0, :]
        y2 = y_temp[:, 1, :]
        y3 = y_temp[:, 2, :]
        y4 = y_temp[:, 3, :]
        y5 = y_temp[:, 4, :]
        y6 = y_temp[:, 5, :]
        y7 = y_temp[:, 6, :]
        y8 = y_temp[:, 7, :]

        yield x_train, [y1, y2, y3, y4, y5, y6, y7, y8]


# Assume black text with white background
def crop(img):
    # https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
    gray = 255 * (img < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w]  # Crop the image - note we do this on the original image
    return rect


def downsize(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
