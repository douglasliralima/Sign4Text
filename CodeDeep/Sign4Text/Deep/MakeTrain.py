import subprocess
from Preprocess import PreProcess
from MyModels import MakeVGG16_pretrained_model, SelfDeepModel

#completed = subprocess.call(['rm model_libras.json'], shell = True)
#completed = subprocess.call(['rm weights_libras.h5'], shell = True)
completed = subprocess.call(['python3 Train.py'], shell = True)

preprocess = PreProcess()
total = preprocess.TotalOfImgs()
Total_Desired = 23000

while(total > Total_Desired):
    print("\n\n\n**********TOTAL OF IMGS FOR TRAINING:", total, "******************\n\n\n")
    completed = subprocess.call(['python3 Train.py'], shell = True)
    total = preprocess.TotalOfImgs()

#p = subprocess.call('python3 Train.py')
#5546