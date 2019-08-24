import socket
import cv2
import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
Traindir = '/home/kafle/Music/sdb1/'
Testdir ='/home/kafle/Pictures/test/'
image_size=600
LR = 1e-3
MODEL_NAME='Plant-identification-convnet'
#HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
#PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#    s.bind((HOST, PORT))
#    s.listen()
#    conn, addr = s.accept()
#    with conn:
#        print('Connected by', addr)
#           data = conn.recv()
#            #convert back the bytearray to image not yet converted
#            img = cv2.imread(data)
#            greyimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#            laplacian = cv2.Laplacian(greyimage,cv2.CV_64F)
#            my_image=cv2.resize(laplacian_array,(image_size,image_size))
#
CATEGORIES=["Acer_Campestre","Acer_Capillipes","Acer_Circinatum"]


for categories in CATEGORIES:
    paths_train=os.path.join(Traindir,categories)
    for image in os.listdir(paths_train):
        image_array_train=cv2.imread(os.path.join(paths_train,image))

training_data=[]
def create_training():
    for categories in CATEGORIES:
        paths_train=os.path.join(Traindir,categories)
        classify=CATEGORIES.index(categories)
        for image in os.listdir(paths_train):
            image_array=cv2.imread(os.path.join(paths_train,image))
            greyimage_array=cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
            laplacian_array = cv2.Laplacian(greyimage_array,cv2.CV_64F)
            my_array=cv2.resize(laplacian_array,(image_size,image_size))
            training_data.append(my_array)
            #training_data.append([np.array(img))
        shuffle(training_data)
        np.save('train_data.npy', training_data)
create_training()
print(len(training_data))




paths_test=os.path.join(Testdir)
for image in os.listdir(paths_test):
    image_array_test=cv2.imread(os.path.join(paths_test,image))

testing_data=[]
def create_testing():
        paths_test=os.path.join(Testdir)
        #classify=CATEGORIES.index(categories)
        for image in os.listdir(paths_test):
            image_array=cv2.imread(os.path.join(paths_test,image))
            greyimage_array=cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
            laplacian_array = cv2.Laplacian(greyimage_array,cv2.CV_64F)
            my_array=cv2.resize(laplacian_array,(image_size,image_size))
            testing_data.append(my_array)
            #training_data.append([np.array(img))
        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
create_testing()
print(len(testing_data))

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
convnet = input_data(shape =[None, image_size, image_size, 1], name ='input')

convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation ='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation ='softmax')
convnet = regression(convnet, optimizer ='adam', learning_rate = LR,
      loss ='categorical_crossentropy', name ='targets')
model = tflearn.DNN(convnet, tensorboard_dir ='log')

train = training_data[:-500]
test = training_data[-500:]
            #check the data
            #feed the input data to the neural network
#X = np.array([i[0] for i in train]).reshape(-1, image_size, image_size, 1)
A = np.array([i[0] for i in train])
X = np.reshape(A, (1,np.product(A.shape)))[0]
Y = [i[1] for i in train]
#test_x = np.array([i[0] for i in test]).reshape(-1, image_size, image_size, 1)
B = np.array([i[0] for i in test])
test_x = np.reshape(A, (1,np.product(B.shape)))[0]
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch = 5,
    validation_set =({'input': test_x}, {'targets': test_y}),
    snapshot_step = 500, show_metric = True, run_id = MODEL_NAME)
model.save(MODEL_NAME)
#for testing my data's
test_data = np.load('test_data.npy')
for num, data in enumerate(test_data):
    model_out = model.predict([data])
    if np.argmax(model_out) == 1: str_label ="Acer_Campestre"
    elif np.argmax(model_out) == 2: str_label ="Acer_Capillipes"
    else : str_label ="Acer_Circinatum"
    y.imshow(orig, cmap ='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
