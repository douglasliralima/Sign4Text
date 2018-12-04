from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D
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
  return matriz


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


imageSize = 64
s_mask = 17


def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis=0).astype(np.float32)


def pretrainedNetwork(xtrain,ytrain,xtest,ytest, model, classweight, numclasses, numepochs, optimizer, labels):
    base_model = model # Topless
    # Add top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Train top layer
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary() #Mostra todas as camadas printando
    # Fit model
    history = model.fit(xtrain,ytrain, batch_size=64, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    # Evaluate model    
    score = model.evaluate(X_test_eva,y_test_evaHot, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test_eva)
    print('\n', sklearn.metrics.classification_report(np.where(y_test_evaHot > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())))
    Y_pred_classes = np.argmax(y_pred,axis = 1)
    Y_true = np.argmax(ytest,axis = 1)
    
    return model

def trainNetwork(xtrain,ytrain,xtest,ytest, model, classweight, numclasses, numepochs, optimizer, labels):    
    # Fit model
    history = model.fit(xtrain,ytrain, batch_size=64, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    # Evaluate model    
    score = model.evaluate(X_test_eva,y_test_evaHot, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test_eva)
    print('\n', sklearn.metrics.classification_report(np.where(y_test_evaHot > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())))
    Y_pred_classes = np.argmax(y_pred,axis = 1)
    Y_true = np.argmax(ytest,axis = 1)
    
    return model

#--------------------------------------------LOAD MODELS TRAIN------------------------------------------
X_train = np.load('ASL_SOBEL_X_RGB32INIT.npy')
y_train = np.load('ASL_SOBEL_Y_RGB32INIT.npy')
X_test_eva = np.load('ASL_SOBEL_X_RGB32_VALI.npy')
y_test_eva = np.load('ASL_SOBEL_Y_RGB32_VALI.npy')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01) 

print(X_train.shape)
from sklearn.utils import shuffle

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)
y_test_evaHot = to_categorical(y_test_eva, num_classes = 30)


model = load_model('Treinamento15epocasRGB_2fit.h5')
confusao = avaliacao(model, X_test_eva, y_test_evaHot)