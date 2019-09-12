'''Import the libraries'''
import pathlib
import random
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

'''Load the dataset'''
data_root = '/home/rishi/Machine_Learning/datasets/shapes'
data_root = pathlib.Path(data_root)

for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

label_names = [item.name for item in data_root.glob('*/') if item.is_dir()]

label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

def preprocess_image(img_raw):
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [28, 28])
    img_final = img_final/255.0
    return img_final

def load_and_preprocess_image(path):
    img_raw = tf.io.read_file(path)
    return preprocess_image(img_raw)

all_images = []

for path in all_image_paths:
    all_images.append(load_and_preprocess_image(path))

all_images= tf.stack(all_images, axis=0)
all_labels = tf.stack(all_image_labels, axis=0)

train_images = all_images[:270]
train_labels = all_labels[:270]

test_images = all_images[270:]
test_labels = all_labels[270:]

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

'''Build the model'''
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

'''Compile the model'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''Train the model'''
model.fit(train_images, train_labels, epochs=20)

'''Evaluate on test set'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)
