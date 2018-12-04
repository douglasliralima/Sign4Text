from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from tqdm import tqdm
from keras.utils import np_utils #Para fazer a categorizaçõa
from sklearn.utils import shuffle #Para bagunçar a base de dados

'''
    A função  abaixo preprocessa uma imagem para deixa-la pronta ao treinamento
    da rede feita pelo próprio keras VGG16
    path     =      Caminho da imagem
'''
class PreProcess:
    '''
        Inicialização da classe, aqui nos definimos certas variáveis que podem
        ser convenientes ao uso do programador
        
        path           =     Caminho da base de dados
        letterNumber   =     Dicionário com a relação entre letras e números na base de dados de libras
    '''
    def __init__(self):
        self.path = os.getcwd() + '/Letras'
        self.letterNumber = {
                            'A': 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5,
                            'G' : 6, 'H' : 7, 'I' : 8, 'J' : 9, 'K' : 10, 'L' : 11,
                            'M' : 12, 'N' : 13, 'O' : 14, 'P' : 15, 'Q' : 16, 'R' : 17,
                            'S' : 18, 'T' : 19, 'U' : 20, 'V' : 21, 'W' : 22, 'X' : 23,
                            'Y' : 24, 'Z' : 25
                            }
        
        
    def preProcessIMGtoVGG16(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    
    def preProcessIMGPattern(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x/255.0
        return x
    
    '''
        A função abaixo pega todas as letras da base de dados que se está sendo usada
        e cria um arquivo correspondente a elas listando-os em uma coluna
    '''
    def makeFileOfImgLetters(self):
        for letter in tqdm(os.listdir(self.path)):
            texto = ''
            for img in os.listdir(self.path + '/' + letter):
                if img.find('.txt') == -1:
                    texto += img + ' ' + '\n'
                else:
                    continue
            arq = open(self.path + '/' + letter + '/' + letter + '.txt', 'w')
            arq.write(texto[:-1])#Para retirar o \n, ele só conta msm como 1 caracter
            arq.close()
    
    '''
        A função abaixo carrega um batch de imagens em memória principal para que
        seja feito o treinamento da VGG16
        path      =      Caminho da base de dados
        N         =      Quantidade de imagens que serão carregadas por letra
        N_thread  =      Quantidade de núcleos do processador
    '''
    def loadBatchLibrasImg(self, N, N_thread):
        #Quantidade de dados que serão usados no treinamento, carregado com a quantidade de imgs que vamos usar
        x = np.zeros((N*26, 224, 224, 3), dtype=np.float32)
        #Array de arrays, sendo que cada array conta com apenas uma posição que é referente a classificação daquela letra
        y = np.zeros((N*26,1), dtype=np.float32)
        i = 0
        #Vamos deixar para cada letra carregar os arquivos no x numpy
        for letter in tqdm(os.listdir(self.path)):
            #Vamos ler seu arquivo associado que já deve ter sido preprocessado
            letter_txt_path = self.path + '/' + letter + '/' + letter + '.txt'
            arq = open(letter_txt_path, 'r')
            texto = arq.read()
            arq.close()
            #Transformamos ele então em um array de strings
            linhas = texto.split('\n')
            texto = [] #Pegamos todo o resto que agora a gente vai usar
            j = 0
            #Depois vamos verificar para cada linha daquilo que foi lido...
            for linha in linhas:
                linha = linha.split(' ')
                #Apenas as linhas que após o split não estão dirty
                if j < N:
                    #if i == 0:
                    #    print("\nImg:", linha[0], "posicao:", i)
                    #Vamos carregar a imagem referente daquela linha na memória
                    
                    x[i] = self.preProcessIMGPattern(self.path + '/' + letter + '/' + linha[0])
                    y[i] = self.letterNumber[letter]
                    #Depois deixamos aquela linha dirty
                    #texto.insert(j, linha[0] + ' x\n')
                    #Ao deixarmos a linha dirty, aumentamos nossos contadores, simbolizando que uma nvoa imagem foi anexada
                    j += 1
                    i += 1
                #Apenas dando o replace no \n no string para salvar no arquivo
                else:
                    #Colocando Apenas as linhas das imagens que não foram loadadas
                    texto.append(linha[0] + ' ' + linha[1] + '\n')
            #Retirando o \n da última linha
            texto[len(texto) - 1] = texto[len(texto) - 1][:-1]
            #print("\nImg:", texto[len(texto) - 1], "posicao:", i - 1)
            arq = open(letter_txt_path, 'w')
            arq.writelines(texto)
            arq.close()
                
        return x, y
    
    '''
        Função utilizada para se saber quantas imagens no total existem no dataset e que estão mapeadas
        nos arquivos txt
    '''
    def TotalOfImgs(self):
        total = 0
        for letter in os.listdir(self.path):
            #Primeira parte é ler o atual arquivo daquela letra
            letter_txt_path = self.path + '/' + letter + '/' + letter + '.txt'
            arq = open(letter_txt_path, 'r')
            texto = arq.read()
            arq.close()
            #Cada linha do arquivo representa uma imagem
            linhas = texto.split('\n')
            #Somamos o tamanho que fica o array
            total += len(linhas)
        return total
    
    def TotalOfImgsLoaded(self):
        
    
    '''
        Função que loada o batch de imagens, categoriza o Y e bagunça tanto o
        X, qnto o Y carregados
        N    =     Numero de imagens a serem carregadas por letra
    '''
    def preProcessData(self, N):
        print(N)
        x_train, y_train = self.loadBatchLibrasImg(N, 12)
        #Categorização do y e mistura dos elementos
        y_train = np_utils.to_categorical(y_train, 26)
        x_train, y_train = shuffle(x_train,y_train, random_state = 0)
        return x_train, y_train

'''
#Códigos para teste no spyder
preprocess = PreProcess()
preprocess.makeFileOfImgLetters()
print("Quantidade total de imagens de libras:", preprocess.TotalOfImgs())
imgs, y = preprocess.loadBatchLibrasImg(100, 12)
img = np.zeros((1,224,224,3), dtype=np.float32)
img[0] = preprocess.preProcessIMGPattern(preprocess.path + '/Q/aug_Q2_372.png')
#Verificação se a imagem tá dentro do big array
        
#Visualização das imagens
import matplotlib.pyplot as plt
plt.imshow(img.reshape(224,224,3))
plt.imshow(imgs[2389].reshape(224,224,3))
'''