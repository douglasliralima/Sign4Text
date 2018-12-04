from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.utils import class_weight
from keras.applications.vgg16 import VGG16
import keras
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import sklearn
import h5py
import os
import cv2
import skimage
from skimage.transform import resize
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from scipy import ndimage


def avaliacao(modelo, atributos_teste, classe_teste):  
  classe_teste = [np.argmax(t) for t in classe_teste]
  #resultado = modelo.evaluate(atributos_teste, classe_teste)
  #print(resultado)  
  classe_previsao = modelo.predict(atributos_teste)
  classe_previsao = [np.argmax(t) for t in classe_previsao]
  matriz = confusion_matrix(classe_previsao, classe_teste)
  print(matriz)

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

class MetricsCheckpoint(Callback):    
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            np.save(self.savepath, self.history)

imageSize = 100

X_train = np.load('ASL_SOBEL_X_32AUG_1CAMADA.npy')
y_train = np.load('ASL_SOBEL_Y_32AUG_1CAMADA.npy')
X_test = np.load('ASL_SOBEL_X_32VALI_1CAMADAS.npy')
y_test = np.load('ASL_SOBEL_Y_32VALI_1CAMADAS.npy')

print(X_train.shape)
from sklearn.utils import shuffle

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01)

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)

map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)


def VGG_19():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(100, 100, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='softmax'))   

    optimizer = keras.optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy']) 

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()

    history = model.fit(X_train,y_trainHot, batch_size=1, epochs=10, validation_data=(X_test,y_testHot), verbose=1, callbacks = [MetricsCheckpoint('logs')])

    X_train2 = np.load('ASL_SOBEL_X_32ORI_1CAMADA.npy')
    y_train2 = np.load('ASL_SOBEL_Y_32ORI_1CAMADA.npy')
    y_trainHot2 = to_categorical(y_train2, num_classes = 30)    
    X_train2, y_trainHot2 = shuffle(X_train2, y_trainHot2, random_state=13)

    history = model.fit(X_train2,y_trainHot2, batch_size=1, epochs=10, validation_data=(X_test,y_testHot), verbose=1, callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(X_test,y_testHot, verbose=0)

    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(xtest)

    print('\n', sklearn.metrics.classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())))
    Y_pred_classes = np.argmax(y_pred,axis = 1)
    Y_true = np.argmax(ytest,axis = 1)

    return model

model = VGG_19()
model.save('/home/lavid-deep/Sign4Text/ModeloASLDuplamenteTreinado10E100SOBELlr0.001.h5')