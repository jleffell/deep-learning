from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.contrib.keras.python.keras.preprocessing.image array_to_img,img_to_array,load_img
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.contrib.keras.python.keras.utils import np_utils
import os
from imutils import paths
import cv2
import argparse
import pickle
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def image_to_feature_vector(image, size=(150, 150)):
    return cv2.resize(image, size).flatten()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
                required=True,
                help="path to input dataset")

ap.add_argument("-w", "--write-data",
                required=False,
                default=False,
                choices=['True', 'False'],
                help="Write data to file")

ap.add_argument("-r", "--read-data",
                required=False,
                default=False,
                choices=['True', 'False'],
                help="Read data from file instead of raw images")

args = ap.parse_args()


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Get Training Data
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args.dataset))

# initialize the data matrix and labels list
data = []
labels = []

if args.write_data:
    print "Selected to write data"

if args.read_data:
    print "Reading data and labels from stored file and not raw images"
    with open("cats_vs_dogs_150.bin", "rb") as data:
        [data, labels] = pickle.load(data)
else:
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        features = image_to_feature_vector(image)
        data.append(features)
        labels.append(label)

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

if args.write_data:
    with open("cats_vs_dogs_150.bin", "wb") as output:
        pickle.dump([data, labels], output)

le = LabelEncoder()
labels = le.fit_transform(labels)


data = np.array(data)
data = np.reshape(data, (data.shape[0], 150, 150, 3))

print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)

batch_size = 16
epochs = 50

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255.0)


# datagen.fit(trainData)

model.fit_generator(train_datagen.flow(trainData,
                                       trainLabels,
                                       batch_size=batch_size),
                    steps_per_epoch=len(trainData) // batch_size,
                    epochs=epochs,
                    validation_data=test_datagen.flow(testData, testLabels,
                                                      batch_size=batch_size),
                    validation_steps=len(testData) // batch_size)
