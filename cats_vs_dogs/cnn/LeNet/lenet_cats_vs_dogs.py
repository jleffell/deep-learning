# import the necessary packages
#import os
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.cnn.networks import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tensorflow.contrib.keras.python.keras.optimizers import SGD, RMSprop
from tensorflow.contrib.keras.python.keras.utils import np_utils
import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from tensorflow.contrib.keras import backend as K
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#K.set_session(sess)


#import cv2

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model",
                type=int,
                required=False,
                default=-1,
                help="Save model to disk")

ap.add_argument("-l", "--load-model",
                type=int,
                required=False,
                default=-1,
                help="Load pre-trained model")

ap.add_argument("-w", "--weights",
                type=str,
                required=False,
                help="Load weights from file")

ap.add_argument("-b", "--batch-size",
                type=int,
                required=False,
                default=128,
                help="Batch size")

ap.add_argument("-e", "--epochs",
                type=int,
                required=False,
                default=20,
                help="Number of epochs")

args = ap.parse_args()

batch_size = args.batch_size
epochs = args.epochs
weightsPath = args.weights if args.load_model > 0 else None

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
#print("[INFO] downloading MNIST...")
#dataset = datasets.fetch_mldata("MNIST Original")

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits

# Read data
with open("cats_vs_dogs_150.bin", "rb") as data:
    [data, labels] = pickle.load(data)

data = np.asarray(data)
print "datashape: ", data.shape
#data = dreshape((dataset.data.shape[0], 32, 32))
data = np.reshape(data, (data.shape[0],150,150,3))
print "after reshape: ", data.shape
#data = data[:, :, :, np.newaxis]

le = LabelEncoder()
labels = le.fit_transform(labels)

data = np.array(data) / 255.0
#labels = np_utils.to_categorical(labels, 2)

(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.2, random_state=42)

#(trainData, validationData, trainLabels, validationLabels) = train_test_split(
#        trainData, trainLabels, test_size=0.2, random_state=42)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)
#validationLabels = np_utils.to_categorical(validationLabels, 2)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
#opt = RMSprop()
model = LeNet.build(width=150,
                    height=150,
                    depth=3,
                    classes=2,
                    weightsPath=weightsPath)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args.load_model < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2)

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=batch_size,
                                      verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args.save_model > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args.weights, overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # resize the image from a 28 x 28 image to a 96 x 96 image so we
    # can better see it
    # image = (testData[i][0] * 255).astype("uint8")
 #   image = (testData[i] * 255).astype("uint8")
 #   image = cv2.merge([image] * 3)
 #   image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
 #   cv2.putText(image, str(prediction[0]), (5, 20),
 #               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
                                                    np.argmax(testLabels[i])))
#    cv2.imshow("Digit", image)
#    cv2.waitKey(0)
