from scipy.ndimage import zoom #Dar o zoom do corte
from PIL import Image
import os


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

#Primeiro vamos navegar até a pasta correta com o auxilio das funções do SO
database = "asl_alphabet_train"
save_folder = "asl_alphabet_train_no_border"
os.mkdir(save_folder)
for letras in os.listdir(database):
    primeira = True
    pastas = database + '/' + letras
    os.mkdir(save_folder + '/' + letras)
    for images in os.listdir(pastas):
        img_path = pastas + '/' + images
        
        #Carregamento da imagem e retransformação da mesma em RGB
        img_file = cv2.imread(img_path, 3)
        b,g,r = cv2.split(img_file)
        img_file = cv2.merge([r,g,b])
        
        #Corta as bordas dando um pequeno zoom nela
        img_file = clipped_zoom(img_file, 1.1)
        
        #Printa a imagem
        print(letras)
        pyplot.imshow(img_file)
        pyplot.show()
        #Salvamos a imagem
        im = Image.fromarray(img_file)
        im.save(save_folder + '/' + letras + '/' + images)
        
        break #Isso é útil para imprimir apenas a primeira imagem e cancelar o for
