import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DR = './train'
TEST_DR = './test'
IMG_SIZE = 50
learningRate = 1e-3 # 0.001

modelName = "dogsvscats-{}-{}.model".format(learningRate, '6conv-basic-15epochs')

# [isCAT,isDOG]
def label_image(img):
    word_label = img.split('.')[0]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DR)):
        label = label_image(img)
        path = os.path.join(TRAIN_DR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    # This is done to randomize the order of cat's and dog's pics
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data

#train_data = create_training_data()
# if the tained data is already saved:
train_data = np.load('training_data.npy', allow_pickle = True)

# https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8
# https://missinglink.ai/guides/convolutional-neural-networks/fully-connected-layers-convolutional-neural-networks-complete-guide/
# Input layer
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# Final output layer
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

# TensorBoard is a visualisation tool kit. Tracking and visualizing metrics such as loss and accuracy.
model = tflearn.DNN(convnet, tensorboard_dir='log')

# Training and testing the labels from the training data set:
train = train_data[:-500] # Everything except the last 500
test = train_data[-500:] # Takes last 500 images for testing

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # The elements in train is a list which corresponds to [image data, label]
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # The elements in train is a list which corresponds to [image data, label]
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=modelName)
# Run these commands on the terminal to access the visual tool kits
# tensorboard --logdir=foo:C:\Users\sudhe\PycharmProjects\CatsvDogs-Flask-TF\log
# tensorboard --logdir=foo:C:\Users\sudhe\PycharmProjects\CatsvDogs-Flask-TF\log --host localhost --port 8088
model.save(modelName)