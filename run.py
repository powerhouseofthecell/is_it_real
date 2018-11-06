import os
import argparse

ap = argparse.ArgumentParser()

ap.add_argument(
    'epochs',
    type=int,
    help='number of epochs to train the model on'
)

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
        epochs=args.epochs,
        validation_data=test_data,
        validation_steps=20,
        use_multiprocessing=False,
        workers=4
    )

def predict(img_dir):
    img_paths = [ os.path.join(img_dir, 'predict/', img_path) for img_path in os.listdir(os.path.join(img_dir, 'predict/')) ]

    images = [ cv2.imread(img) for img in img_paths ]
    images = [ cv2.resize(img, (im_height, im_width)) for img in images ]
    images = [ np.reshape(img, [1, im_height, im_width, 3]) for img in images ]

    return [ (m.predict_classes(img)[0][0], img_paths[i]) for i, img in enumerate(images) ]

def invert_mapping(d):
    inverted = dict()

    for key, value in d.items():
        inverted[value] = key
    
    return inverted

try:
    if __name__ == '__main__':
        train_data, test_data = load_data('img')
        build_model()
        train_model(train_data, test_data)

        predictions = predict('img')

        mapping = invert_mapping(train_data.class_indices)

        for val, im_name in predictions:
            print(f'We think that {im_name} is {mapping[val]}')

except KeyboardInterrupt:
    print('\nUser aborted!')
