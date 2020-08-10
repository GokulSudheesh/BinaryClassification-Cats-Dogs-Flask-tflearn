import tflearn
from model import return_networks
import numpy as np
from tqdm import tqdm
import os
import cv2
import time

# Loading the pre-trained model
net = return_networks()
model = tflearn.DNN(net)
model.load('dogsvscats-0.001-6conv-basic-15epochs.model')# This pre-trained model has 86% accuracy.
IMG_SIZE = 50
TEST_DR = './test'
def test():
    # A function which simply tests new data from the test directory
    for img in tqdm(os.listdir(TEST_DR)):
        path = os.path.join(TEST_DR, img)
        frame = cv2.imread(path)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        img_data = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict(img_data)[0] # First element from the list (list within a list)
        str_label = ''
        if np.argmax(model_out) == 1: # Returns the index of the largest element in the list
            str_label = 'Dog'
        else:
            str_label = 'Cat'
        cv2.putText(frame, str_label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("classifier", frame)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

def recognize(path):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
    img_data = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict(img_data)[0]  # First element from the list (list within a list)
    str_label = ''
    if np.argmax(model_out) == 1:  # Returns the index of the largest element in the list
        str_label = 'Dog'
    else:
        str_label = 'Cat'
    return (str_label)
#test()
#print(recognize("C:\\Users\\sudhe\\Downloads\\test3.jpeg"))