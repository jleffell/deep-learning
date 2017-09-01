#
#   Based on tutorial at pyimagesearch.com
#
#   http://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/
#

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Activation
from tensorflow.contrib.keras.python.keras.optimizers import SGD
from tensorflow.contrib.keras.python.keras.layers.core import Dense
from tensorflow.contrib.keras.python.keras.utils import np_utils
from tensorflow.contrib.keras.python.keras import backend as K
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Resize images
def image_to_feature_vector(image, size=(150, 150)):
    '''
    resize the image to a fixed size, then flatten the image into
    a list of raw pixel intensities
    '''
    return cv2.resize(image, size).flatten()


# Construct the argument parse and parse the arguments
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

ap.add_argument("-b", "--batch-size",
                required=False,
                default=32,
                type=int, 
                help="Read data from file instead of raw images")

ap.add_argument("-e", "--epochs",
                required=False,
                default=20,
                type=int,
                help="Read data from file instead of raw images")

args = ap.parse_args()

# Constants
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
IMG_WIDTH, IMG_HEIGHT = 150, 150

#if K.image_data_format() == 'channels_first':
#        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
#else:
#        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args.dataset))

# initialize the data matrix and labels list
data = []
labels = []

if args.read_data:
    print "Reading data and labels from stored file and not raw images"
    with open("cats_vs_dogs.bin", "rb") as data:
        [data, labels] = pickle.load(data)
else:      
    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        # construct a feature vector raw pixel intensities, then update
        # the data matrix and labels list
        features = image_to_feature_vector(image)
        data.append(features)
        labels.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

if args.write_data:
    with open("cats_vs_dogs.bin","wb") as output:
        pickle.dump([data, labels], output)
        
# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072,
                kernel_initializer="glorot_uniform",
                activation="relu"))
model.add(Dense(384, kernel_initializer="glorot_uniform",
                activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
model.summary()
model.fit(trainData, trainLabels, epochs=EPOCHS, batch_size=BATCH_SIZE,
          verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
                                  batch_size=BATCH_SIZE, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                                                     accuracy * 100))
