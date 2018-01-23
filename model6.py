# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-100 dataset classification task.
Inspired by reference:
    Let’s keep it simple, Using simple architectures to outperform deeper and more complex architectures
    Seyyed Hossein Hasanpour, Mohammad Rouhani, Mohsen Fayyaz, Mohammad Sabokrou
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization as batch_norm
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn import optimizers 
import tensorflow as tf
import numpy as np


# Data loading and preprocessing
from tflearn.datasets import cifar100
(X, Y), (X_test, Y_test) = cifar100.load_data(dirname='../cifar-100-python')
X, Y = shuffle(X, Y)
#Y = to_categorical(Y, 100)
#Y_test = to_categorical(Y_test, 100)

num_classes = 100
print( 'X', X.shape)
print( 'Y', Y.shape)
Y_test = np.array(Y_test)

train_ind = np.load('train-ind.pkl')
val_ind = np.load('val-ind.pkl')
print('train ind', train_ind.shape)
print('val ind', val_ind.shape)

num_train = len(train_ind)
num_val = len(val_ind)
print('num train', num_train)
print('num val', num_val)

X_train = X[train_ind]
X_val = X[val_ind]
print('Xtrain', X_train.shape)
print('Xval', X_val.shape)

# create one hot label matrices
one_hot = np.zeros((len(Y), 100))
ind = np.arange(len(Y))
one_hot[ind, Y[ind]] = 1.0

one_hot_train = one_hot[train_ind]
one_hot_val = one_hot[val_ind]

one_hot_test = np.zeros((len(Y_test), 100))
ind_test = np.arange(len(Y_test))
one_hot_test[ind_test, Y_test[ind_test]] = 1.0


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

with tf.Session() as sess:
  tflearn.is_training(True)

# Convolutional network building
# ARCHITECUTRE
net = input_data(shape=(None, 32, 32, 3),
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
net = conv_2d(net, 64, 3, activation='relu')
net = conv_2d(net, 128, 3, activation='relu')
net = dropout(net, .5)
net = max_pool_2d(net, 2)
net = conv_2d(net, 128, 3, activation='relu')
net = conv_2d(net, 128, 3, activation='relu')
net = dropout(net, .5)
net = max_pool_2d(net, 2)
net = conv_2d(net, 128, 3, activation='relu')
net = dropout(net, .5)
net = max_pool_2d(net, 2)
net = conv_2d(net, 128, 3, activation='relu')
net = dropout(net, .5)

net = fully_connected(net, 512, activation='relu')
net = dropout(net, .5)
net = fully_connected(net, 100, activation='softmax')
#sgd = optimizers.SGD(learning_rate=0.0001, lr_decay=0.00005)
net = regression(net, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Train using classifier
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='./tmp/tflearn_logs/')

# Load already trained model
model.load('./cifar_model6.tflearn')

# TRAINING
# uncomment and change n_epoch for training
#model.fit(X_train, one_hot_train, n_epoch=100, snapshot_epoch=True, snapshot_step=200, 
#          shuffle=True, validation_set=(X_val, one_hot_val), 
#          show_metric=True, batch_size=100, run_id='cifar100_cnn_model6')

#model.save('./cifar_model6.tflearn')


# EVALUATION
# time to evaluate
with tf.Session() as sess:
  tflearn.is_training(False)

metrics = model.evaluate(X_test, one_hot_test)
print('metrics', metrics)

#predict_test = model.predict_label(X_test)

#pickle.dump(Y_test, open('true-test-labels.pkl', 'wb'))
#pickle.dump(predict_test, open('predict-test-labels.pkl', 'wb'))

