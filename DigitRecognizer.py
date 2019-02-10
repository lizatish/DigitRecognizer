#!/usr/bin/env python3
# По материалам интернета:
# https://habrahabr.ru/post/321834/
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Отключение предупреждений и информационных сообщений по уровням.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Отключение предупреждений по категориям.
import warnings
warnings.filterwarnings(action = 'ignore', category = FutureWarning)

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Параметры.
train_data_size = 3000
validate_data_size = 210
batch_size = 16
epochs = 20
image_size = 35
# Включён режим тестирование без обучения.
test_only = True
# Включён режим пошагового обучения с загрузкой файла весов, полученных на предыдущем шаге.
next_training_phase = False
# Блокировка первого свёрточного слоя.
lock_convolution_layer1 = False
# Блокировка второго свёрточного слоя.
lock_convolution_layer2 = False
# Блокировка третьего свёрточного слоя.
lock_convolution_layer3 = False
# Блокировка четвёртого свёрточного слоя.
lock_convolution_layer4 = False
# Блокировка пятого свёрточного слоя.
lock_convolution_layer5 = False
# Блокировка шестого свёрточного слоя.
lock_convolution_layer6 = False
# Блокировка полносвязных слоёв.
lock_dense_part = False
tests_number = 20

# Модель.
model = Sequential()
# filters - сколько вариации пикселя будет из одного пикселя
# kernel - сколько обовляется пикселей одновременно с исходным
# padding - вид обработки краев
# input_shape (size, size, кол-во каналов (rgb))
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'valid',
    input_shape = (image_size, image_size, 3)))
# Вызов функции активации вида relu
model.add(Activation('relu'))
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'valid'))
model.add(Activation('relu'))
# Сужение изображения в 2 раза
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'valid'))
model.add(Activation('relu'))
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Convolution2D(filters = 64, kernel_size = (3, 3), padding = 'valid'))
# model.add(Activation('relu'))
# model.add(Convolution2D(filters = 64, kernel_size = (3, 3), padding = 'valid'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
# Кол-во нейронов на relu
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Кол-во нейронов на softmax (10 папок) - softmax для распознавания хорошо
model.add(Dense(10))
model.add(Activation('softmax'))
data_generator = ImageDataGenerator(rescale = 1.0 / 255)
if test_only:
    print('Testing only.')
else:
    if next_training_phase:
        print('Continued training.')
        model.load_weights('best_model.hdf5')
    else:
        print('The starting phase of training.')
    if lock_convolution_layer1:
        print('Locking of the first convolution layer.')
        for layer in model.layers[0:2]:
            layer.trainable = False
    if lock_convolution_layer2:
        print('Locking of the second convolution layer.')
        for layer in model.layers[2:5]:
            layer.trainable = False
    if lock_convolution_layer3:
        print('Locking of the third convolution layer.')
        for layer in model.layers[5:7]:
            layer.trainable = False
    if lock_convolution_layer4:
        print('Locking of the fourth convolutional layer.')
        for layer in model.layers[7:10]:
            layer.trainable = False
    if lock_convolution_layer5:
        print('Locking of the fifth convolutional layer.')
        for layer in model.layers[10:12]:
            layer.trainable = False
    if lock_convolution_layer6:
        print('Locking of the sixth convolutional layer.')
        for layer in model.layers[12:15]:
            layer.trainable = False
    if lock_dense_part:
        print('Locking dense layers.')
        for layer in model.layers[15:]:
            layer.trainable = False
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
if not test_only:
    # Обучение.
    train_generator = data_generator.flow_from_directory('data/train',
        target_size = (image_size, image_size), batch_size = batch_size, class_mode = 'categorical')
    validation_generator = data_generator.flow_from_directory('data/validation',
        target_size = (image_size, image_size), batch_size = batch_size, class_mode = 'categorical')
    model.fit_generator(train_generator, callbacks = [ModelCheckpoint('best_model.hdf5',
        monitor = 'val_acc', save_best_only = True, save_weights_only = False, mode = 'auto')],
        steps_per_epoch = train_data_size // batch_size, epochs = epochs,
        validation_data = validation_generator, validation_steps = validate_data_size // batch_size)
# Тестирование.
model.load_weights('best_model.hdf5')
test_generator = data_generator.flow_from_directory('data/test',
    target_size = (image_size, image_size), batch_size = tests_number, class_mode = 'categorical')
model.evaluate_generator(test_generator, steps = tests_number)
data, labels = test_generator.next()
images = np.asarray([img_to_array(item) for item in data])
predictions = np.asarray(model.predict(data))
samples = [sample for sample in zip(images, predictions, labels)]
plt.figure(figsize = (10, 8))
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace = 0.2, hspace = 0.2)
for index, sample in enumerate(samples[:tests_number]):
    plt.subplot(4, 5, index + 1)

    image = sample[0]

    max_value = 0.0
    prediction = 0
    for i in range(10):
        value = sample[1][i]
        if (value > max_value):
            prediction = i
            max_value = value

    max_value = 0.0
    max_label = 0
    for i in range(10):
        value = sample[2][i]
        if (value > max_value):
            max_label = i
            max_value = value

    if(prediction == max_label):
        plt.text(0, 0, max_label, fontsize=11, color='green')
        plt.text(30, 0, prediction, fontsize=11, color='green')
    else:
        plt.text(0, 0, max_label, fontsize=11, color='blue')
        plt.text(30, 0, prediction, fontsize=11, color='red')

    plt.imshow(image)
    plt.axis('off')
plt.gcf().canvas.set_window_title('Первые 20 результатов тестирования')
plt.show()

