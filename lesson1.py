'''Import the libraries'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

'''Load the dataset'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''Preprocess the data'''
train_images = train_images/255.0
test_images = test_images/255.0

'''Build the model'''
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

'''Compile the model'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''Train the model'''
model.fit(train_images, train_labels, epochs = 5)

'''Evaluate on test set'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)

'''Make a single prediction'''
img = test_images[0]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
print(np.argmax(predictions_single))
