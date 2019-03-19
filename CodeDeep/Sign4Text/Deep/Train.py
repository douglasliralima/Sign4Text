#Classe com uma série de funções úteis para o preprocessamento de libras
from Preprocess import PreProcess
from MyModels import MakeVGG16_pretrained_model, SelfDeepModel, MobileNetModel
import os
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


#Para saber se o PC está usando bem a GPU, utilizar o comando:
'''nvidia-smi dmon'''

'''
    Função simples que apenas vai criar o modelo de Deep Learning
    aug_A1_411.png
    A2_346.png
'''

class MetricsCheckpoint(Callback):    
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            np.save(self.savepath, self.history)


def ConfusionMatrix(modelo, atributos_teste, classe_teste):
  
  classe_teste = [np.argmax(t) for t in classe_teste]
  #resultado = modelo.evaluate(atributos_teste, classe_teste)
  #print(resultado)
  
  classe_previsao = modelo.predict(atributos_teste)
  classe_previsao = [np.argmax(t) for t in classe_previsao]
  matriz = confusion_matrix(classe_previsao, classe_teste)
  return matriz

def makeModel():
    model = MakeVGG16_pretrained_model((150,150,3), 26)
    return model

def saveModelAndWeights(model, modelName):
    #Podemos então carregar todo o classificador pelo json em string com um simples método
    model_json = model.to_json()
    with open(modelName + '.json', 'w') as json_file:
        json_file.write(model_json)
    #Depois salvamos os pesos de nossa rede
    #No computador é necessário fazer um pip install h5py
    model.save_weights("Weights_" + modelName + '.h5')


#Se o modelo já existe:
preprocess = PreProcess()
model = None
modelName = "model_librasMobileNetIMGNET"
if os.path.isfile(modelName + ".json"):
    print("\n-----------------------------Loading Model----------------------------------\n")
    model = PreProcess.loadModelAndWeights(modelName)
else:
    print("\n-------------------making file with img of letter names------------------------\n")
    preprocess.makeFileOfImgLetters()
    print("\n----------------------------Making Model-----------------------------\n")
    model = makeModel()
model.summary()

x_train, y_train = preprocess.preProcessData(1200)

otimizador = Adam(lr = 0.0001, decay = 0.00000)
    
model.compile(loss = 'categorical_crossentropy', optimizer = otimizador, metrics = ['accuracy'])

#preprocess.makeFileOfImgLetters(True)
x_test, y_test = preprocess.preProcessData(1200, True)
    
model.fit(x_train, y_train, batch_size = 64, epochs = 10, validation_split=0.05)

#model.fit(x_train, y_train, batch_size = 32, epochs = 5, validation_data=(x_test,y_test), callbacks = [MetricsCheckpoint('logs')])

'''
y_pred = model.predict(x_test)

report = classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1), target_names=list(preprocess.map_characters.values()))


avaliacao = model.evaluate(x_train, y_train)
matriz = ConfusionMatrix(model, x_train, y_train)
print(matriz)
'''

saveModelAndWeights(model, modelName)

'''
x_train = None
y_train = None

import gc
gc.collect()

os.mkdir("")
'''
