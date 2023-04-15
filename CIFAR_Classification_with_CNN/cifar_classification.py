#Simple CNN with CIFAR Dataset

#import libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Download and prepare CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#initialize the list of class names in dataset
class_names = ['airplane',
               'automobile',
               'bird',
               'cat',
               'deer',
               'dog',
               'frog',
               'horse',
               'ship',
               'truck']

#plot 25 images from each class
plt.figure(figsize=(10,10))

#for loop to plots

for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i])

  #extra index for CIFAR lables
  plt.xlabel(class_names[train_labels[i][0]])

#show the images
plt.show()

#Create Convolutional Base for Model
Conv2D and MaxPooling2D layers

model = models.Sequential()

#add layers for 32
#add Conv2D layer
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (32, 32, 3)))
#add MaxPooling layer
model.add(layers.MaxPooling2D((2,2)))

#add layers for 64
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

model.summary()

#Add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation= 'relu'))
model.add(layers.Dense(10))

#check the model summary
model.summary()

Model training

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True ),
              metrics = ['accuracy'])

#fit the model
model_fit = model.fit(train_images, train_labels, epochs=10,
                      validation_data = (test_images, test_labels))

**Model Evaluation**

plt.plot(model_fit.history['accuracy'], label='accuracy')
plt.plot(model_fit.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
