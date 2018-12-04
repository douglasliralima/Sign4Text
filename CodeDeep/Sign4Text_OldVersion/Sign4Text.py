
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

'''
def get_data(path1, path2):
  
    X = []
    y = []  
                    
               
    #Caminho2
    for people in os.listdir(path2):
        if people in ['A']:
          for folderName in os.listdir(path2 + people):
            if not folderName.startswith('.'):
                if folderName in ['a']:
                    label = 0
                elif folderName in ['b']:
                    label = 1
                elif folderName in ['c']:
                    label = 2
                elif folderName in ['d']:
                    label = 3
                elif folderName in ['e']:
                    label = 4
                elif folderName in ['f']:
                    label = 5
                elif folderName in ['g']:
                    label = 6
                elif folderName in ['h']:
                    label = 7
                elif folderName in ['i']:
                    label = 8
                elif folderName in ['j']:
                    label = 9
                elif folderName in ['k']:
                    label = 10
                elif folderName in ['l']:
                    label = 11
                elif folderName in ['m']:
                    label = 12
                elif folderName in ['n']:
                    label = 13
                elif folderName in ['o']:
                    label = 14
                elif folderName in ['p']:
                    label = 15
                elif folderName in ['q']:
                    label = 16
                elif folderName in ['r']:
                    label = 17
                elif folderName in ['s']:
                    label = 18
                elif folderName in ['t']:
                    label = 19
                elif folderName in ['u']:
                    label = 20
                elif folderName in ['v']:
                    label = 21
                elif folderName in ['w']:
                    label = 22
                elif folderName in ['x']:
                    label = 23
                elif folderName in ['y']:
                    label = 24
                elif folderName in ['z']:
                    label = 25
                elif folderName in ['del']:
                    label = 26
                elif folderName in ['nothing']:
                    label = 27
                elif folderName in ['space']:
                    label = 28           
                else:
                    label = 29
                for image_filename in tqdm(os.listdir(path2 + people + '/' + folderName)):
                    img_file = cv2.imread(path2 + people + '/' + folderName + '/' + image_filename, cv2.IMREAD_COLOR)                    
                    if img_file is not None:                                                
                        #img_file = rgb2lab(img_file / 255.0)[:,:,0]                         
                        sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
                        sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
                        sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
                        sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
                        img_file = 0.5 * sobelx + 0.5 * sobely                        
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')
                        img_arr = np.asarray(img_file.astype(np.float16))
                        X.append(img_arr.astype(np.float16))
                        y.append(label)
                        
                        
                      
                      
                      
        if people in ['B']:
          for folderName in os.listdir(path2 + people):
            if not folderName.startswith('.'):
                if folderName in ['a']:
                    label = 0
                elif folderName in ['b']:
                    label = 1
                elif folderName in ['c']:
                    label = 2
                elif folderName in ['d']:
                    label = 3
                elif folderName in ['e']:
                    label = 4
                elif folderName in ['f']:
                    label = 5
                elif folderName in ['g']:
                    label = 6
                elif folderName in ['h']:
                    label = 7
                elif folderName in ['i']:
                    label = 8
                elif folderName in ['j']:
                    label = 9
                elif folderName in ['k']:
                    label = 10
                elif folderName in ['l']:
                    label = 11
                elif folderName in ['m']:
                    label = 12
                elif folderName in ['n']:
                    label = 13
                elif folderName in ['o']:
                    label = 14
                elif folderName in ['p']:
                    label = 15
                elif folderName in ['q']:
                    label = 16
                elif folderName in ['r']:
                    label = 17
                elif folderName in ['s']:
                    label = 18
                elif folderName in ['t']:
                    label = 19
                elif folderName in ['u']:
                    label = 20
                elif folderName in ['v']:
                    label = 21
                elif folderName in ['w']:
                    label = 22
                elif folderName in ['x']:
                    label = 23
                elif folderName in ['y']:
                    label = 24
                elif folderName in ['z']:
                    label = 25
                elif folderName in ['del']:
                    label = 26
                elif folderName in ['nothing']:
                    label = 27
                elif folderName in ['space']:
                    label = 28           
                else:
                    label = 29
                for image_filename in tqdm(os.listdir(path2 + people + '/' + folderName)):
                    img_file = cv2.imread(path2 + people + '/' + folderName + '/' + image_filename, cv2.IMREAD_COLOR)
                    if img_file is not None:                                                
                        #img_file = rgb2lab(img_file / 255.0)[:,:,0]                                       
                        sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
                        sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
                        sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
                        sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
                        img_file = 0.5 * sobelx + 0.5 * sobely                        
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')
                        img_arr = np.asarray(img_file.astype(np.float16))
                        X.append(img_arr.astype(np.float16))
                        y.append(label)
                        
                        
                      
        if people in ['C']:
          for folderName in os.listdir(path2 + people):
            if not folderName.startswith('.'):
                if folderName in ['a']:
                    label = 0
                elif folderName in ['b']:
                    label = 1
                elif folderName in ['c']:
                    label = 2
                elif folderName in ['d']:
                    label = 3
                elif folderName in ['e']:
                    label = 4
                elif folderName in ['f']:
                    label = 5
                elif folderName in ['g']:
                    label = 6
                elif folderName in ['h']:
                    label = 7
                elif folderName in ['i']:
                    label = 8
                elif folderName in ['j']:
                    label = 9
                elif folderName in ['k']:
                    label = 10
                elif folderName in ['l']:
                    label = 11
                elif folderName in ['m']:
                    label = 12
                elif folderName in ['n']:
                    label = 13
                elif folderName in ['o']:
                    label = 14
                elif folderName in ['p']:
                    label = 15
                elif folderName in ['q']:
                    label = 16
                elif folderName in ['r']:
                    label = 17
                elif folderName in ['s']:
                    label = 18
                elif folderName in ['t']:
                    label = 19
                elif folderName in ['u']:
                    label = 20
                elif folderName in ['v']:
                    label = 21
                elif folderName in ['w']:
                    label = 22
                elif folderName in ['x']:
                    label = 23
                elif folderName in ['y']:
                    label = 24
                elif folderName in ['z']:
                    label = 25
                elif folderName in ['del']:
                    label = 26
                elif folderName in ['nothing']:
                    label = 27
                elif folderName in ['space']:
                    label = 28           
                else:
                    label = 29
                for image_filename in tqdm(os.listdir(path2 + people + '/' + folderName)):
                    img_file = cv2.imread(path2 + people + '/' + folderName + '/' + image_filename, cv2.IMREAD_COLOR)
                    if img_file is not None:                                              
                        #img_file = rgb2lab(img_file / 255.0)[:,:,0]                                 
                        sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
                        sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
                        sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
                        sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
                        img_file = 0.5 * sobelx + 0.5 * sobely                        
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')
                        img_arr = np.asarray(img_file.astype(np.float16))
                        X.append(img_arr.astype(np.float16))
                        y.append(label)
                        
                        
                      
                      
                      
        if people in ['D']:
          for folderName in os.listdir(path2 + people):
            if not folderName.startswith('.'):
                if folderName in ['a']:
                    label = 0
                elif folderName in ['b']:
                    label = 1
                elif folderName in ['c']:
                    label = 2
                elif folderName in ['d']:
                    label = 3
                elif folderName in ['e']:
                    label = 4
                elif folderName in ['f']:
                    label = 5
                elif folderName in ['g']:
                    label = 6
                elif folderName in ['h']:
                    label = 7
                elif folderName in ['i']:
                    label = 8
                elif folderName in ['j']:
                    label = 9
                elif folderName in ['k']:
                    label = 10
                elif folderName in ['l']:
                    label = 11
                elif folderName in ['m']:
                    label = 12
                elif folderName in ['n']:
                    label = 13
                elif folderName in ['o']:
                    label = 14
                elif folderName in ['p']:
                    label = 15
                elif folderName in ['q']:
                    label = 16
                elif folderName in ['r']:
                    label = 17
                elif folderName in ['s']:
                    label = 18
                elif folderName in ['t']:
                    label = 19
                elif folderName in ['u']:
                    label = 20
                elif folderName in ['v']:
                    label = 21
                elif folderName in ['w']:
                    label = 22
                elif folderName in ['x']:
                    label = 23
                elif folderName in ['y']:
                    label = 24
                elif folderName in ['z']:
                    label = 25
                elif folderName in ['del']:
                    label = 26
                elif folderName in ['nothing']:
                    label = 27
                elif folderName in ['space']:
                    label = 28           
                else:
                    label = 29
                for image_filename in tqdm(os.listdir(path2 + people + '/' + folderName)):
                    img_file = cv2.imread(path2 + people + '/' + folderName + '/' + image_filename, cv2.IMREAD_COLOR)
                    if img_file is not None:                                          
                        #img_file = rgb2lab(img_file / 255.0)[:,:,0]                                          
                        sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
                        sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
                        sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
                        sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
                        img_file = 0.5 * sobelx + 0.5 * sobely                        
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')
                        img_arr = np.asarray(img_file.astype(np.float16))
                        X.append(img_arr.astype(np.float16))
                        y.append(label)              
                        

                      
                      
                      
                      
        if people in ['E']:
          for folderName in os.listdir(path2 + people):
            if not folderName.startswith('.'):
                if folderName in ['a']:
                    label = 0
                elif folderName in ['b']:
                    label = 1
                elif folderName in ['c']:
                    label = 2
                elif folderName in ['d']:
                    label = 3
                elif folderName in ['e']:
                    label = 4
                elif folderName in ['f']:
                    label = 5
                elif folderName in ['g']:
                    label = 6
                elif folderName in ['h']:
                    label = 7
                elif folderName in ['i']:
                    label = 8
                elif folderName in ['j']:
                    label = 9
                elif folderName in ['k']:
                    label = 10
                elif folderName in ['l']:
                    label = 11
                elif folderName in ['m']:
                    label = 12
                elif folderName in ['n']:
                    label = 13
                elif folderName in ['o']:
                    label = 14
                elif folderName in ['p']:
                    label = 15
                elif folderName in ['q']:
                    label = 16
                elif folderName in ['r']:
                    label = 17
                elif folderName in ['s']:
                    label = 18
                elif folderName in ['t']:
                    label = 19
                elif folderName in ['u']:
                    label = 20
                elif folderName in ['v']:
                    label = 21
                elif folderName in ['w']:
                    label = 22
                elif folderName in ['x']:
                    label = 23
                elif folderName in ['y']:
                    label = 24
                elif folderName in ['z']:
                    label = 25
                elif folderName in ['del']:
                    label = 26
                elif folderName in ['nothing']:
                    label = 27
                elif folderName in ['space']:
                    label = 28           
                else:
                    label = 29
                for image_filename in tqdm(os.listdir(path2 + people + '/' + folderName)):
                    img_file = cv2.imread(path2 + people + '/' + folderName + '/' + image_filename, cv2.IMREAD_COLOR)
                    if img_file is not None:                                                
                        #img_file = rgb2lab(img_file / 255.0)[:,:,0]                                             
                        sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
                        sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
                        sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
                        sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
                        img_file = 0.5 * sobelx + 0.5 * sobely                        
                        img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')   
                        img_arr = np.asarray(img_file.astype(np.float16))
                        X.append(img_arr.astype(np.float16))
                        y.append(label)           
                        
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
                    sobelx = np.abs(cv2.Sobel(img_file, cv2.CV_64F, 1, 0, ksize=s_mask))
                    sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
                    sobely = np.abs(cv2.Sobel(img_file,cv2.CV_64F,0,1,ksize=s_mask))
                    sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)
                    img_file = 0.5 * sobelx + 0.5 * sobely                    
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3), mode='reflect')                         
                    img_arr = np.asarray(img_file.astype(np.float16))
                    X.append(img_arr.astype(np.float16))
                    y.append(label)
 
    
    X = np.asarray(X).astype(np.float16)
    y = np.asarray(y)
    return X,y

train_dir1 = "asl_alphabet_train_augmentation/"
train_dir2 = "dataset5_augmentation/"
X_train, y_train = get_data(train_dir1, train_dir2)

np.save("ASL_SOBEL_X_RGB16AUG.npy", X_train)
np.save("ASL_SOBEL_Y_RGB16AUG.npy", y_train)

'''
#dataset = read_dataset('dataset', N_CLASSES, len(PERSON_FOLDERS), RESIZED_IMAGE)

X_train = np.load('ASL_SOBEL_X_RGB32INIT.npy')
y_train = np.load('ASL_SOBEL_Y_RGB32INIT.npy')
X_test_eva = np.load('ASL_SOBEL_X_RGB32_VALI.npy')
y_test_eva = np.load('ASL_SOBEL_Y_RGB32_VALI.npy')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01) 

print(X_train.shape)
from sklearn.utils import shuffle

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)

map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}  

#cnn_model = load_model('/home/lavid-deep/Sign4Text/ModeloASLDuplamenteTreinadoLAB.h5')
#avaliacao(cnn_model, X_test, y_testHot)

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)

map_characters1 = map_characters
class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
weight_path1 = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = VGG16(weights = weight_path1, include_top=False, input_shape=(imageSize, imageSize, 3))
print("foi")
optimizer1 = keras.optimizers.Adam()
 
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
    model.summary()
    # Fit model
    history = model.fit(xtrain,ytrain, batch_size=64, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    # Evaluate model    
    score = model.evaluate(X_test_eva,y_test_eva, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test_eva)
    print('\n', sklearn.metrics.classification_report(np.where(y_test_eva > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())))
    Y_pred_classes = np.argmax(y_pred,axis = 1)
    Y_true = np.argmax(ytest,axis = 1)
    
    return model

def pretrainedNetwork2(xtrain,ytrain,xtest,ytest, model, classweight, numclasses, numepochs, optimizer, labels):    
    # Fit model
    history = model.fit(xtrain,ytrain, batch_size=64, epochs=numepochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    # Evaluate model    
    score = model.evaluate(X_test_eva,y_test_eva, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(X_test_eva)
    print('\n', sklearn.metrics.classification_report(np.where(y_test_eva > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())))
    Y_pred_classes = np.argmax(y_pred,axis = 1)
    Y_true = np.argmax(ytest,axis = 1)
    
    return model
  
print("ouie")
model = pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot, model, class_weight1, 30, 15, optimizer1, map_characters1)

X_train = np.load('ASL_SOBEL_X_RGB32GEN.npy')
y_train = np.load('ASL_SOBEL_Y_RGB32GEN.npy')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01) 

print(X_train.shape)
from sklearn.utils import shuffle

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)

X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)

model = pretrainedNetwork2(X_train, y_trainHot, X_test, y_testHot, model, class_weight1, 30, 15, optimizer1, map_characters1)

model.save('Treinamento15epocasRGB_2fit.h5')