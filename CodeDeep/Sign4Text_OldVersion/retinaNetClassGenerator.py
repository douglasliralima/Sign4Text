from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr

#Um mapeamento legal que foi indicado é fazer dois dicionários um para as partições de treinamento e teste
#E outro falando dos valores de labels
partition = {
        'train' : ['id-1', 'id-2', 'id-3'],
        'validation': ['id-4']
        }

labels = {
        'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1
        }

map_characters = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E', 
        5: 'F', 
        6: 'G', 
        7: 'H', 
        8: 'I', 
        9: 'J', 
        10: 'K',
        11: 'L', 
        12: 'M', 
        13: 'N', 
        14: 'O', 
        15: 'P', 
        16: 'Q', 
        17: 'R', 
        18: 'S', 
        19: 'T', 
        20: 'U', 
        21: 'V', 
        22: 'W', 
        23: 'X', 
        24: 'Y',
        25: 'Z'}

#Toda classe generator recebe como parâmetro um objeto da classe keras.utils.Sequence que chamamos de generator
#Esse objeto vai funcionar como uma classe pai e vamos implementar seu filho

class Sign4TextGenerator(keras.utils.Sequence):
        def __init__( self, data_dir, set_name, classes = map_characters, image_extension='.jpg', 
                     skip_truncated=False, skip_difficult=False, shuffle=True, **kwargs):
            
            self.data_dir             = data_dir
            self.set_name             = set_name
            self.classes              = classes
            self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
            self.image_extension      = image_extension
            self.skip_truncated       = skip_truncated
            self.skip_difficult       = skip_difficult
            self.shuffle              = shuffle
            
            #Basicamente vamos pegar as nomeaçoes da classe e salvar nessas labels
            self.labels = {}
            for key, value in self.classes.items():
                self.labels[value] = key
            
            super(PascalVocGenerator, self).__init__(**kwargs)
            
       def size(self):
            """ Size of the dataset.
            """
            return len(self.image_names)
    
        def num_classes(self):
            """ Number of classes in the dataset.
            """
            return len(self.classes)
    
        def has_label(self, label):
            """ Return True if label is a known label.
            """
            return label in self.labels
    
        def has_name(self, name):
            """ Returns True if name is a known class.
            """
            return name in self.classes
    
        def name_to_label(self, name):
            """ Map name to label.
            """
            return self.classes[name]
    
        def label_to_name(self, label):
            """ Map label to name.
            """
            return self.labels[label]
        
        
        def name_to_label(self, name):
            """ Map name to label.
            """
            return self.classes[name]
    
        def label_to_name(self, label):
            """ Map label to name.
            """
            return self.labels[label]


        def image_aspect_ratio(self, image_index):
            """ Compute the aspect ratio for an image with image_index.
            """
            path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
            image = Image.open(path)
            return float(image.width) / float(image.height)
        
        
        def load_image(self, image_index):
            """ Load an image at the image_index.
            """
            path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
            return read_image_bgr(path)