from Preprocess import PreProcess
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import numpy as np

'''
    A função abaixo retorna uma matriz
'''
def ConfusionMatrix(modelo, atributos_teste, classe_teste):
  
  classe_teste = [np.argmax(t) for t in classe_teste]
  #resultado = modelo.evaluate(atributos_teste, classe_teste)
  #print(resultado)
  
  classe_previsao = modelo.predict(atributos_teste)
  classe_previsao = [np.argmax(t) for t in classe_previsao]
  matriz = confusion_matrix(classe_previsao, classe_teste)
  return matriz


preprocess = PreProcess()
modelName = "model_librasMobileNetIMGNET"
model = PreProcess.loadModelAndWeights(modelName)
otimizador = Adam(lr = 0.0003, decay = 0.00001)
model.compile(loss = 'categorical_crossentropy', optimizer = otimizador, metrics = ['accuracy'])
model.summary()

preprocess.makeFileOfImgLetters(True)
x_train, y_train = preprocess.preProcessData(300, True)
avaliacao = model.evaluate(x_train, y_train)
print(avaliacao)
matriz = ConfusionMatrix(model, x_train, y_train)

'''
import matplotlib.pyplot as plt
plt.imshow(x_train[2])
plt.show()
'''