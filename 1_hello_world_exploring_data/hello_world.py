"""
Training URL from : https://www.tensorflow.org/tutorials/keras/basic_classification
Author: Chris teo
Project: Hello world, project to explore tensorflow
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Loading data for mnist
fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#create list from training
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Image scaling for smaller datasets
train_images = train_images / 255.0
test_images = test_images / 255.0


#Exploring datasets (Plotting data to see what it looks like) #1
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#Exploring datasets (Plotting data to see what it looks like) #2
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

#Keras, a high level API that defines neural networks
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #28x28 --> 784
    keras.layers.Dense(128, activation=tf.nn.relu), #layer with 128 nodes
    keras.layers.Dense(10, activation=tf.nn.softmax) #10 node softmax Normalise to a prob function
])

#Compiler definition
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model
model.fit(train_images, train_labels, epochs=5)

#Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#Utilising the model to predict
predictions = model.predict(test_images)



#Data representation
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Evaluate datas
i=0
while i!=-1:
    print ("====Input '-1' to exit=====")
    print ("test for image number:\n")
    i = int(input("Image= ")) #Play around with i to test images
    if (i<0):
        break
    else:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(1,2,2)
        plot_value_array(i, predictions,  test_labels)
        plt.show()
        i=0
#Comment to see git change
