import cv2
import socket
import os
import sqlite3
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
Traindir = '/home/kafle/Music/sdb1/'
Testdir ='/home/kafle/Pictures/test/'
CATEGORIES=["Acer_Campestre","Acer_Capillipes","Acer_Circinatum"]
image_size=500
LR = 1e-3
MODEL_NAME='Plant-identification-convnet'

#preparing Traning train_data
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
            my_array=cv2.resize(laplacian_array,(image_size,image_size),3)
            training_data.append(my_array)
            #training_data.append([np.array(img))
        shuffle(training_data)
        np.save('train_data.npy', training_data)
create_training()
print(len(training_data))


#for Static testing
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
            my_array=cv2.resize(laplacian_array,(image_size,image_size),3)
            testing_data.append(my_array)
            #training_data.append([np.array(img))
        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
create_testing()
print(len(testing_data))
test = np.load("test_data.npy")
print("[STATUS] feature vector size {}".format(np.array(test).shape))
X_test = np.array([i[0] for i in test]).reshape(-1, image_size, image_size, 1)
print(len(X_test))
Y_test = [i[1] for i in test]


train = np.load("train_data.npy")
print("[STATUS] feature vector size {}".format(np.array(train).shape))
X_train=np.array([i[0] for i in train]).reshape(-1, image_size, image_size, 1)
Y_train = [i[1] for i in train]



from keras.utils import np_utils
new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_train/= 255
new_X_test/=255
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

model = Sequential()
model.add(Conv2D(500, (3, 3), input_shape=(image_size, image_size, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(new_X_train, new_Y_train, epochs=10, batch_size= 500)

import h5py
model.save('Trained_model.h5')

from keras.models import load_mode
HOST = '127.0.10.1'
PORT = 65432
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        data = conn.recv(4096)
        data = str(data)
        img = cv2.imread(data)
        image_array = np.array(img)
        image_array = image_array.astype('float32')
        #image_array /= 255.0
        image_array = image_array.reshape(1, image_size, image_size, 1)
        model = load_model('Trained_model.h5')
        answer = model.predict(image_array)
        input_image.show()
        model_outname=labels[np.argmax(answer)]
        print(model_outname)
        name=mode_outname
        conn = sqlite3.connect("/home/kafle/Plant_ident_ification")
        mycursor = conn.cursor()

        medicalvalue=mycursor.execute("SELECT medicalValue FROM Plant_identification WHERE name='parash'")
        fetchmedicalvalue = mycursor.fetchone()
        print(fetchmedicalvalue)
        bytfetchmedicalvalue=fetchmedicalvalue.encode()
        s.send(bytfetchmedicalvalue)

        topologicalRegion=mycursor.execute("SELECT topologicalRegion FROM Plant_identification WHERE name='parash'")
        fetchtopologicalRegion = mycursor.fetchone()
        print(fetchtopologicalRegion)
        bytfetchtopologicalRegion=fetchtopologicalRegion.encode()
        s.send(bytfetchtopologicalRegion)

        humidity=mycursor.execute("SELECT humidity FROM Plant_identification WHERE name='parash'")
        fetchhumidity = mycursor.fetchone()
        print(fetchhumidity)
        bytfetchhumidity=fetchhumidity.encode()
        s.send(bytfetchhumidity)


        usefulParts=mycursor.execute("SELECT usefulParts FROM Plant_identification WHERE name='parash'")
        fetchusefulParts = mycursor.fetchone()
        print(fetchusefulParts)
        bytfetchusefulParts=fetchusefulParts.encode()
        s.send(bytfetchusefulParts)

        image=mycursor.execute("SELECT image FROM Plant_identification WHERE name='parash'")
        fetchimage = mycursor.fetchone()
        print(fetchimage)
        bytfetchimage=fetchimage.encode()
        s.send(bytfetchimage)

        conn.close()

    s.close()
