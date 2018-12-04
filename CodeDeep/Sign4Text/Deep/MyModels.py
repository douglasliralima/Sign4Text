from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Add, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet


'''
    Função que cria um modelo personalizado da VGG16 no uso de Deep Learning
    img_shape     =     Dimensões da imagem, como (224,224,3) como uma imagem 224x224 em RGB
    num_classes   =     Número das classes que serão preditas pelo modelo
'''

def MakeVGG16_pretrained_model(img_shape, num_classes, imagenet = True):
    if imagenet == True:
        print("=================USING VGG16 WITH IMGNET====================")
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
    else:
        print("=================USING VGG16 WITH RANDOM VALUES====================")
        model_vgg16_conv = VGG16(include_top=False, input_shape=img_shape)
    #model_vgg16_conv.summary()
    
    #Create your own input format
    #keras_input = Input(shape=img_shape, name = 'image_input')
    
    #Use the generated model 
    #output_vgg16_conv = model_vgg16_conv(keras_input)
    
    #Add the fully-connected layers 
    x = model_vgg16_conv.output
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)   
    
    #Create your own model 
    pretrained_model = Model(inputs=model_vgg16_conv.input, outputs=x)
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
    
    
    return pretrained_model

'''
    Função que cria um modelo personalizado da VGG16 no uso de Deep Learning
    img_shape     =     Dimensões da imagem, como (224,224,3) como uma imagem 224x224 em RGB
    num_classes   =     Número das classes que serão preditas pelo modelo
'''
def SelfDeepModel(img_shape, num_classes):
    #Camada entrada
    camadaEntrada = Input(shape = img_shape)
    #Camada oculta
    x = Conv2D(32, (2,2), strides = (2,2), padding = 'same', activation = 'relu')(camadaEntrada)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    for i in range(6):
      x = Conv2D(64, (4,4), strides = (2,2), padding = 'same', activation = 'relu')(x)
      x = BatchNormalization()(x)
      x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)
    
      
    x = Conv2D(64, (4,4), strides = (3,3), padding = 'same', activation = 'relu')(camadaEntrada)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    
    x = Flatten()(x)
    #Camada saída
    
    x = Dense(units = 254, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units = 508, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units = 1016, activation = 'relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(units = 1016, activation = 'relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(units = 1016, activation = 'relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(units = 508, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units = 128, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(units = num_classes, activation = 'softmax')(x)
    
    return Model(inputs = [camadaEntrada], outputs = [x])


def MobileNetModel(img_shape, num_classes, imagenet = True):
    if imagenet == True:
        print("=================USING MOBILENET WITH IMGNET====================")
        mobile = MobileNet(weights='imagenet', include_top=False, input_shape=img_shape)
    else:
        print("=================USING MOBILENET WITH RANDOM VALUES====================")
        mobile = MobileNet(include_top=False, input_shape=img_shape)
    #model_vgg16_conv.summary()
    
    #Create your own input format
    #keras_input = Input(shape=img_shape, name = 'image_input')
    
    #Use the generated model 
    #output_vgg16_conv = model_vgg16_conv(keras_input)
    
    #Add the fully-connected layers 
    x = mobile.output
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)   
    
    #Create your own model 
    pretrained_model = Model(inputs=mobile.input, outputs=x)
    for layer in mobile.layers:
        layer.trainable = False
    
    
    return pretrained_model

