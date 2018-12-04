import subprocess
import os

#p = subprocess.call('ffmpeg -i A1.mp4 -vf scale=224:224 -r 20 AFT/A1_%04d.png')
#path1 é onde fica os videos originais
path1 = '/home/lavid/Documentos/Sign4Text/VideosLibras'
#path2 é a pasta destino das fotos
path2 = '/home/lavid/Documentos/Sign4Text/Deep/LetrasValidacao'
print(os.listdir(path1))
print(os.listdir(path2))

i = 0
for folderName in os.listdir(path1):
    for  i in range(7, 8):
        print("ffmpeg -i " +path1+ "/" + str(folderName) + "/" + str(folderName) + str(i) +".mp4 -vf scale=224:224 -r 20 " +path2  +"/"+ str(folderName) + "/" + str(folderName) + str(i) +"_%04d.png") 
        p = subprocess.call("ffmpeg -i " +path1+ "/" + str(folderName) + "/" + str(folderName) + str(i) +".mp4 -vf scale=224:224 -r 20 " + path2 +"/"+ str(folderName) + "/" + str(folderName) + str(i) +"_%04d.png", shell = True)              