import os
import csv

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def adjust_brightness(img, value=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def adjust(img, brightness=-100, contrast=30):
    res = np.copy(img)
    res = np.int16(res)
    res = res * (contrast / 127 + 1) - contrast + brightness
    res = np.clip(res, 0, 255)
    res = np.uint8(res)
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
                img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
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
                img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                steering_left = steering_center + correction
                images.append(img_left)
                angles.append(steering_left)

                img_right = cv2.imread(os.path.join(sample_dir, 'IMG',
                                                    batch_sample[2].split(os.sep)[-1]))
                img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                steering_right = steering_center - correction
                images.append(img_right)
                angles.append(steering_right)

                for img, angle in [(img_center, steering_center),
                                   (img_left, steering_left),
                                   (img_right, steering_right)]:
                    img_adjusted = adjust_brightness(img, value=np.random.uniform(low=0.2, high=1.0))
                    images.append(img_adjusted)
                    angles.append(angle)

                for img, angle in [(img_center, steering_center),
                                   (img_left, steering_left),
                                   (img_right, steering_right)]:
                    x1 = np.random.randint(low=-50, high=50)
                    x2 = np.random.randint(low=30, high=80)
                    img_adjusted = adjust(img, brightness=x1, contrast=x2)
                    images.append(img_adjusted)
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def network(train_samples, validation_samples, train_generator, validation_generator, batch_size=32):
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
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])

    print(model.summary())

    #model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
    # nb_val_samples=len(validation_samples), nb_epoch=2)
    history = model.fit_generator(train_generator, verbose=1, validation_data=validation_generator,
                        epochs=3, steps_per_epoch=len(train_samples) / batch_size,
                        validation_steps=len(validation_samples) / batch_size)
    model.save('model.h5')
    return history


def plot_distribution(samples):
    fig = plt.figure()
    angles = [25 * float(sample[3]) for sample in samples]
    hist, bin_edges = (np.histogram(angles, bins=[a for a in range(-25, 26, 1)]))
    plt.bar(bin_edges[:-1], hist, width=1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.xticks([a for a in range(-26, 27, 4)])
    #plt.show()
    fig.savefig('examples/samples_distribution.png')


def plot_loss(history):
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    fig.savefig('examples/model_loss.png')


def plot_accuracy(history):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    fig.savefig('examples/model_accuracy.png')


def main():
    # gpu memory settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    KTF.set_session(tf.Session(config=config))

    sample_dir = 'data/dataset2'
    samples = load_sample(sample_dir)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    batch_size = 16
    train_generator = generator(train_samples, batch_size=batch_size, sample_dir=sample_dir)
    validation_generator = generator(validation_samples, batch_size=batch_size, sample_dir=sample_dir)
    history = network(train_samples, validation_samples, train_generator, validation_generator, batch_size=batch_size)

    plot_distribution(samples)
    #print(history.history.keys())
    plot_loss(history)
    plot_accuracy(history)


if __name__ == '__main__':
    main()

