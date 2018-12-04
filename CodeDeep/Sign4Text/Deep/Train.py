#Classe com uma série de funções úteis para o preprocessamento de libras
from Preprocess import PreProcess
from MyModels import MakeVGG16_pretrained_model, SelfDeepModel, MobileNetModel
import os
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import confusion_matrix

#Para saber se o PC está usando bem a GPU, utilizar o comando:
'''nvidia-smi dmon'''

'''
    Função simples que apenas vai criar o modelo de Deep Learning
    aug_A1_411.png
    A2_346.png
'''
def ConfusionMatrix(modelo, atributos_teste, classe_teste):
  
  classe_teste = [np.argmax(t) for t in classe_teste]
  #resultado = modelo.evaluate(atributos_teste, classe_teste)
  #print(resultado)
  
  classe_previsao = modelo.predict(atributos_teste)
  classe_previsao = [np.argmax(t) for t in classe_previsao]
  matriz = confusion_matrix(classe_previsao, classe_teste)
  return matriz

def makeModel():
    model = MobileNetModel((224,224,3), 26, True)
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

x_train, y_train = preprocess.preProcessData(800)

otimizador = Adam(lr = 0.0001, decay = 0.00000)
    
model.compile(loss = 'categorical_crossentropy', optimizer = otimizador, metrics = ['accuracy'])
    
model.fit(x_train, y_train, batch_size = 64, epochs = 10, validation_split=0.05)

saveModelAndWeights(model, modelName)

#avaliacao = model.evaluate(x_train, y_train)
#matriz = ConfusionMatrix(model, x_train, y_train)
#print(matriz)

'''
x_train = None
y_train = None

import gc
gc.collect()

os.mkdir("")
'''
