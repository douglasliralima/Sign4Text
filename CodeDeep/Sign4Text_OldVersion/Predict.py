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

imageSize = 64
s_mask = 17

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def olhafunc(indice):
    if indice == 0:
        return 'A'
    elif indice == 1:
        return 'B'
    elif indice == 2:
        return 'C'
    elif indice == 3:
        return 'D'
    elif indice == 4:
        return 'E'
    elif indice == 5:
        return 'F'
    elif indice == 6:
        return 'G'
    elif indice == 7:
        return 'H'
    elif indice == 8:
        return 'I'
    elif indice == 9:
        return 'J'
    elif indice == 10:
        return 'K'
    elif indice == 11:
        return 'L'
    elif indice == 12:
        return 'M'
    elif indice == 13:
        return 'N'
    elif indice == 14:
        return 'O'
    elif indice == 15:
        return 'P'
    elif indice == 16:
        return 'Q'
    elif indice == 17:
        return 'R'
    elif indice == 18:
        return 'S'
    elif indice == 19:
        return 'T'
    elif indice == 20:
        return 'U'
    elif indice == 21:
        return 'V'
    elif indice == 22:
        return 'W'
    elif indice == 23:
        return 'X'
    elif indice == 24:
        return 'Y'
    elif indice == 25:
        return 'Z'
    
def avaliacao(modelo, atributos_teste, classe_teste):
  
  classe_teste = [np.argmax(t) for t in classe_teste]
  #resultado = modelo.evaluate(atributos_teste, classe_teste)
  #print(resultado)
  
  classe_previsao = modelo.predict(atributos_teste)
  classe_previsao = [np.argmax(t) for t in classe_previsao]
  matriz = confusion_matrix(classe_previsao, classe_teste)
  print(matriz)
  

cnn_model = load_model('ModeloLibrasRGB20EAUG.h5')
imagens = []
for image_filename in tqdm(os.listdir("ASL/E/")):
    img_file = cv2.imread("ASL/E/" + image_filename, 0)

    if img_file is not None:
        #img_file = rgb2lab(img_file / 255.0)[:,:,0]
        #sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
        #sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
        #sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
        #sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
        #img_file = 0.5 * sobelx + 0.5 * sobely
                  
        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')        
        img_arr = np.asarray(img_file.astype(np.float32))
        print(img_arr.shape)
        img_arr = img_arr.reshape((1, 64, 64, 3))
        
        print("\n" * 2)
        print(cnn_model.predict(img_arr))
        print("\n" * 2)
        print(olhafunc(np.argmax(cnn_model.predict(img_arr))))
        
        

