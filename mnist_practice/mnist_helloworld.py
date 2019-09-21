"""
resource:
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
https://www.tensorflow.org/tutorials/estimators/cnn

"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time


#Import dataset for processing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#plot image
def plotImage(x_train,y_train,image_index): # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    plt.imshow(x_train[image_index], cmap='Greys')
    plt.show()

#plotImage(x_train,y_train,10)
###Learning the data
#print (x_train.shape) #(60000, 28, 28)

#Preprocessing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #6000(4D) x 28(3D) x 28(2D) x 1(1D)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) #6000(4D) x 28(3D) x 28(2D) x 1(1D)
input_shape = (28, 28, 1)
x_train =x_train/ 255
x_test =x_test/ 255

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

#CNN networks using keras : https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
def make_cnn_network(train_images,train_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)
    return model
model=make_cnn_network(x_train,y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)
