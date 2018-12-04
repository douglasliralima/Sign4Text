
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


imageSize = 64
s_mask = 17


def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis=0).astype(np.float32)

def get_data(path1):
  
    X = []
    y = []                
            
                        
        #Caminho1
    for folderName in os.listdir(path1):      
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in tqdm(os.listdir(path1 + folderName)):
                img_file = cv2.imread(path1 + folderName + '/' + image_filename, cv2.IMREAD_COLOR)
                if img_file is not None:                                        
                    #img_file = rgb2lab(img_file / 255.0)[:,:,0] 
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')                         
                    img_arr = np.asarray(img_file.astype(np.float32))
                    X.append(img_arr.astype(np.float32))
                    y.append(label)
 
    
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y)
    return X,y

train_dir1 = "ASL/"
X_train, y_train = get_data(train_dir1)

np.save("ASL_SOBEL_X_RGB32", X_train)
np.save("ASL_SOBEL_Y_RGB32", y_train)