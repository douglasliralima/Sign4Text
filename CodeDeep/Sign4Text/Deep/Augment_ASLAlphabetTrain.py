from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm

def get_picture(img_path):
    img_file = cv2.imread(img_path, 3)
    b,g,r = cv2.split(img_file)
    img_file = cv2.merge([r,g,b])
    return img_file


gerador_treinamento = ImageDataGenerator(rotation_range = 10,
                                         #horizontal_flip = True,
                                         shear_range = 3,
                                         height_shift_range = 0.001, #Variação vertical, vai de -x~x
                                         width_shift_range = 0.01, #variação horizontal
                                         zoom_range = 0.2,
                                         horizontal_flip = True,
                                         #rescale = 1.3,
                                         channel_shift_range = 50
                                        )

database = "Letras"

for letters in os.listdir(database):
    print("Letra:", letters)
    letters_path = database + '/' + letters
    for img in tqdm(os.listdir(letters_path)):  
            if ".txt" in img:
                continue
            img_path = letters_path + '/' + img
            img_file = get_picture(img_path)
            
            heigh, width, channel = img_file.shape            
            j = 0
            img_file = img_file.reshape(1, heigh, width, channel) #Transformando em rank 4
            #Geração da imagem e salvamento
            for X_batch in gerador_treinamento.flow(img_file, batch_size = 3):
                #Organizando em rank 3 para plottar
                X_batch = X_batch.reshape(heigh, width, channel)
                
                im = Image.fromarray(X_batch.astype('uint8'))
                im.save(letters_path + '/' + 'Aug_' + img)
                break
print("Geração Finalizada")