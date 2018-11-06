import os
import argparse

ap = argparse.ArgumentParser()

args = ap.parse_args()

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import cv2
import numpy as np

m = Sequential()
im_height, im_width = 150, 150

def load_data(img_dir):
    # train_imgs_paths = [ os.path.join(img_dir, 'train/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'train/')) ]
    # test_imgs_paths = [ os.path.join(img_dir, 'test/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'test/')) ]

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            os.path.join(img_dir, 'train/'),
            target_size=(im_height, im_width),
            batch_size=20,
            class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
            os.path.join(img_dir, 'test/'),
            target_size=(im_height, im_width),
            batch_size=20,
            class_mode='binary'
    )

    return train_generator, validation_generator
    
def build_model():
    m.add(Conv2D(64, (3, 3), input_shape=(im_height, im_width, 3), activation='relu'))
    
    m.add(MaxPooling2D(pool_size=(5, 5)))

    m.add(Flatten())

    m.add(Dense(32, activation='sigmoid'))
    m.add(Dropout(0.4))
    m.add(Dense(32, activation='sigmoid'))

    m.add(Dense(1, activation='sigmoid'))

    m.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )

def train_model(train_data, test_data):
    m.fit_generator(
        train_data,
        steps_per_epoch=20,
        epochs=10,
        validation_data=test_data,
        validation_steps=20,
        use_multiprocessing=False,
        workers=4
    )

try:
    if __name__ == '__main__':
        train_data, test_data = load_data('img')
        build_model()
        train_model(train_data, test_data)

except KeyboardInterrupt:
    print('\nUser aborted!')
