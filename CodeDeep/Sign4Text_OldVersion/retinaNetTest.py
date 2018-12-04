#Bibliotecas referentes a retinanet
from keras_retinanet import models, losses
from keras.optimizers import Adam
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import tensorflow as tf

#Bibliotecas usadas como auxiliar
import numpy as np
from keras.utils.np_utils import to_categorical


#Para carregar a rede
model = models.backbone('resnet50').retinanet(num_classes=27)
print(model.summary())

#Para a compilação do nosso modelo
model.compile(
    loss={
        'regression'    : losses.smooth_l1(),
        'classification': losses.focal()
    },
    optimizer = Adam(lr=1e-5, clipnorm=0.001)
)

#Carregamento
X_train = np.load('ASL_SOBEL_X_RGB32INIT.npy')
y_train = np.load('ASL_SOBEL_Y_RGB32INIT.npy')
X_test_eva = np.load('ASL_SOBEL_X_RGB32_VALI.npy')
y_test_eva = np.load('ASL_SOBEL_Y_RGB32_VALI.npy')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01) 

print(X_train.shape)

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)

map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}  

#cnn_model = load_model('/home/lavid-deep/Sign4Text/ModeloASLDuplamenteTreinadoLAB.h5')
#avaliacao(cnn_model, X_test, y_testHot)

#Muito grande para shuffla
#X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=13)
#X_test, y_testHot = shuffle(X_test, y_testHot, random_state=13)
generator = (X_train, y_trainHot)

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly para ver como faz esse fit_generator
model.fit_generator(generator, steps_per_epoch = 200, epochs = 10)
