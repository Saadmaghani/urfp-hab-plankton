import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from training import EarlyStopping, FocalLoss
from torchvision import transforms
from preprocessing import Rescale, RandomCrop, ToTensor

"""
versions:
1.x - defaults + just images
2.x - adam optimizer
3.x - early stopping
4.x - thresholding
5.x - focal loss + proportional reduction of classes
5.3.0 - focal loss + thresholding (test against 4.x) (rename to 5.3.0)
6.x - focal loss + proportional reduction + minimum 
7.x - focal loss + data augmentation + thresholding. 
8.x - 16 random crops

strategies (preprocessing):
1.0 - "thresholding" | thresholding
2.0 - "propReduce" | proportional reduction
2.1 - "propReduce_min" | proportional reduction with lower bound 
3.0 - "augmentation" | data augmentation
3.1 - "augmentation_max" | data augmentation with upper bound  

strategies (training):
0.0 - Transfer Learning
1.0 - "FocalLoss" | Focal loss
2.0 - optim.Adam | Adam optimizer
3.0 - EarlyStopping | Early stopping with Patience = 20
4.0 - 16 random crops and averaging the output
4.1 - 16 random crops, not averaging output, making into one layer and then feeding into FC
5.0 - autoencoder

"""

# default:
# lr = 0.01, optim - optim.SGD, epochs = 10, es = None, loss_fun = nn.MSELoss, momentum = 0.9, batch size = 128, es w/ patience=10

# versions correspond to hyperparameters.
# version 1.0 = 500 images
# version 1.1 = 2000 images
# version 1.2 = 1000 images
# version 2.0 = lr - 0.0003, optim - adam, 500 images
# version 3.0 = lr - 0.0003, optim - adam, 500 images, es = Early Stopping w/ patience 10, epochs = 200
# version 3.1 = same as 3.0 except minibatch_size = 256
# version 3.2 = further training of HP 3.1, model 2.2
# version 3.3 = further training of HP 3.1, model 2.2, patience = 20
# version 3.4 = same as 3.1 + lr_scheduler w/ StepLR & step size = 7
# version 3.5 = same as 3.1 except patience = 20
# version 3.6 = same as 3.5 except patience = 40
# version 3.7 = same as 3.5 except 1000 images ***
# version 4.0 = same as 3.5 except 2000 images and thresholding
# version 4.1 = same as 4.0 except 2500 images
# version 5.0 = same as 3.5 except maxN = 30000, no thresholding, no images/class, loss_fc = FocalLoss
# version 5.1 = same as 5.0 except maxN = 56000 which is similar N to 4.1 (56111)
# version 5.2 = same as 5.0 except maxN = 100000
# version 5.3.0 = same as 4.1 except with focal loss. to test against threshold w/o focal loss.
# version 6.0 = same as 5.0 except minimum = 100. this minimum means that if there are less than min images, include all st n = min(N, minimum). This will remove the population dist. bias but accuracies might increase. lets see
# version 6.1 = same as 6.0 except minimum = 200
# version 6.2 = same as 6.0 except minimum = 300
# version 6.3 = same as 6.0 except minimum = 400
# version 6.4 = same as 6.0 except minimum = 500
# version 6.5 = same as 6.0 except minimum = 600
# version 7.0 = thresholding + data augmention Test with "augmentation"=T, "thresholding"=T, "number_of_images_per_class"=200 and "minimum"=100. no MaxN, same as 3.0 
# version 7.1 = same as 7.0 except minimum = 400. #images/class = 2500. tests data augmentation and thresholding.
# version 8.0 = same as 7.1 transformations added. can only work with GoogleNet 2.0, minibatch size = 64
# version 8.1 = same as 8.0 to test the above methods lets keep a standard of 1000 images per class.
# version 9.0 = transformations but batch_size = 1 and 16 random crops work as minibatch. only work with GoogleNet 3.0
# version 10.0 = same as 4.0 except batch_size = 50, 100 images and training the autoencoder. so transforms is with rescale to (128, 264)
# version 10.1 = same as 10.0 except batch_Size = 256, 1000 images, EarlyStopping(patience=20,mode='min') 
class Hyperparameters:
    version=10.1
    learning_rate = 0.0003
    number_of_epochs = 200
    momentum = 0.9
    optimizer = optim.Adam
    loss_function = nn.MSELoss
    es = EarlyStopping(patience=20, mode='min')
    batch_size = 256
    scheduler = None
    pp_strategy = "thresholding"
    maxN = None 
    minimum = None
    number_of_images_per_class = 1000
    transformations = transforms.Compose([Rescale((128, 264)), ToTensor()]) #transforms.Compose([RandomCrop(16), Rescale((64, 128), multiple=True), ToTensor(multiple=True)])





