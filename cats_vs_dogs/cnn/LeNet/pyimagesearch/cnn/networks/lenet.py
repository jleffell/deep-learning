# import the necessary packages
from tensorflow.contrib.keras.python.keras.models import Model, Sequential
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.core import Activation, Dropout
from tensorflow.contrib.keras.python.keras.layers.core import Flatten
from tensorflow.contrib.keras.python.keras.layers.core import Dense
from tensorflow.contrib.keras.python.keras import regularizers

rp = 0.0

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=(height, width, depth),kernel_regularizer=regularizers.l2(rp)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same",kernel_regularizer=regularizers.l2(rp)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


        # second set of CONV => RELU => POOL
        model.add(Conv2D(256, (3, 3), padding="same",kernel_regularizer=regularizers.l2(rp)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # second set of CONV => RELU => POOL
#        model.add(Conv2D(512, (3, 3), padding="same",kernel_regularizer=regularizers.l2(rp)))
#        model.add(Activation("relu"))
#        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
#        model.add(Conv2D(512, (5, 5), padding="same",
#                         kernel_regularizer=regularizers.l2(0.01),
#                         ))
#        model.add(Activation("relu"))
#        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        
        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256,kernel_regularizer=regularizers.l2(rp)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        # set of FC => RELU layers
#        model.add(Dense(500,kernel_regularizer=regularizers.l2(0.01)))
#        model.add(Activation("relu"))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
