'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation
examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
# from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.optimizers import SGD, RMSprop
from tensorflow.contrib.keras.python.keras import regularizers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'

nb_train_samples = 20000
nb_validation_samples = 2500
nb_test_samples = 2500
epochs = 50
batch_size = 32
data_augmentation = False

kernel_initializer = 'glorot_uniform'
kernel_regularizer = None # regularizers.l2(0.0)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(8, (9, 9), input_shape=input_shape, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (7, 7), kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(250, (3, 3), kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(125, activation='relu', kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


sgd = SGD(lr=0.01, momentum=0.9)
# rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()

# this is the augmentation configuration we will use for training
if data_augmentation :
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.0,
        zoom_range=0.0,
        rotation_range=10.0,
        horizontal_flip=False)
else:
    train_datagen = ImageDataGenerator(
                rescale=1. / 255)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=nb_test_samples,
    class_mode='categorical')

# Extract test data
testData, testLabels = test_generator.next()

# Evaluate test data
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=1)

# Report accuracy on test data
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
