import os
import csv

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from skimage import transform

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def brightness(img, value=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def rotate(img, angle=5.0):
    res = transform.rotate(img, angle=angle, resize=False, mode='edge')
    res *= 255
    return res


def rescale(img, scale=1.0):
    res = transform.rescale(img, scale=scale, mode='constant')
    res = res[0:160, 0:320]
    res *= 255
    return res


def to_yuv(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return res


def load_sample(sample_dir):
    lines = []
    with open(os.path.join(sample_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for index, line in enumerate(reader):
            if index == 0:
                continue
            lines.append(line)
    return lines


def generator(samples, sample_dir='.', batch_size=32):
    num_samples = len(samples)
    # loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img_center = cv2.imread(os.path.join(sample_dir, 'IMG',
                                                     batch_sample[0].split(os.sep)[-1]))
                images.append(img_center)
                steering_center = float(batch_sample[3])
                correction = 0.20
                angles.append(steering_center)

                # flipped
                image_flipped = np.fliplr(img_center)
                images.append(image_flipped)
                measurement_flipped = -steering_center
                angles.append(measurement_flipped)

                # left, right cameras
                img_left = cv2.imread(os.path.join(sample_dir, 'IMG',
                                                   batch_sample[1].split(os.sep)[-1]))
                steering_left = steering_center + correction
                images.append(img_left)
                angles.append(steering_left)

                img_right = cv2.imread(os.path.join(sample_dir, 'IMG',
                                                    batch_sample[2].split(os.sep)[-1]))
                steering_right = steering_center - correction
                images.append(img_right)
                angles.append(steering_right)

                # brightness adjustments
                for img, angle in [(img_center, steering_center), (img_left, steering_left), (img_right, steering_right)]:
                    for k in range(10):
                        img_adjusted = brightness(img, value=np.random.uniform(low=0.2, high=1.0))
                        images.append(img_adjusted)
                        angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def custom(train_samples, validation_samples, train_generator, validation_generator, batch_size=32):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Conv2D(25, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.8))
    model.add(Dense(50))
    model.add(Dropout(0.9))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, verbose=1, validation_data=validation_generator,
                        epochs=1, steps_per_epoch=len(train_samples) / batch_size,
                        validation_steps=len(validation_samples) / batch_size)
    model.save('model.h5')


def main():
    # adjusting tf memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    KTF.set_session(tf.Session(config=config))

    sample_dir = 'data/dataset1'
    samples = load_sample(sample_dir)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    batch_size = 16
    train_generator = generator(train_samples, batch_size=batch_size, sample_dir=sample_dir)
    validation_generator = generator(validation_samples, batch_size=batch_size, sample_dir=sample_dir)
    custom(train_samples, validation_samples, train_generator, validation_generator, batch_size=batch_size)


if __name__ == '__main__':
    main()

