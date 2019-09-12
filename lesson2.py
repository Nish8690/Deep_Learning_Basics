'''Import the libraries'''
import tensorflow as tf
from tensorflow import keras

'''Load the dataset'''
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

'''Preprocess the data'''
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images/255.0, test_images/255.0

'''Build the model'''
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

'''Compile the model'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''Train the model'''
model.fit(train_images, train_labels, epochs=5)

'''Evaluate on test set'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)
