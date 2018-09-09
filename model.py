import os
import csv

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D


def load_data(dirs):
    images = []
    measurements = []
    for image_dir in dirs:
        lines = []
        basedir = image_dir
        print('Processing %s' % basedir)
        with open(os.path.join(basedir, 'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for index, line in enumerate(reader):
                if index == 0:
                    continue
                lines.append(line)

        print('Training images %d' % len(lines))
        for line in lines:
            img_center = cv2.imread(os.path.join(basedir, 'IMG', line[0].split('/')[-1]))
            images.append(img_center)

            steering_center = float(line[3])
            correction = 0.3  # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            measurements.append(steering_center)

            # flipped
            image_flipped = np.fliplr(img_center)
            images.append(image_flipped)
            measurement_flipped = -steering_center
            measurements.append(measurement_flipped)

            # left, right cameras
            img_left = cv2.imread(os.path.join(basedir, 'IMG', line[1].split('/')[-1]))
            steering_left = steering_center + correction
            images.append(img_left)
            measurements.append(steering_left)

            img_right = cv2.imread(os.path.join(basedir, 'IMG', line[2].split('/')[-1]))
            steering_right = steering_center - correction
            images.append(img_right)
            measurements.append(steering_right)

        print('Augmented images %d' % len(images))
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


def lenet(X_train, y_train):
    model = Sequential()

    # Normalizing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # Cropping the image
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    model.add(Conv2D(6, kernel_size=(5, 5)))
    model.add(Activation('relu'))

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #  Layer 2: Convolutional. Output = 10x10x16.
    model.add(Conv2D(16, kernel_size=(5, 5)))
    model.add(Activation('relu'))

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten. Input = 5x5x16. Output = 400.
    model.add(Flatten())

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    # Layer 5: Fully Connected. Input = 84. Output = 1.
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    model.save('model.h5')


def custom(X_train, y_train):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(25, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.9))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    model.save('model.h5')


if __name__ == '__main__':
    dirs = ['data/dataset1', 'data/dataset2']
    X_train, y_train = load_data(dirs)
    custom(X_train, y_train)
