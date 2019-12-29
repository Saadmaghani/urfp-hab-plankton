import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from training import EarlyStopping, FocalLoss, VAE_Criterion
from torchvision import transforms
from preprocessing import Rescale, RandomCrop, ToTensor, Normalize

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
# version 4.0 = same as 3.5 except 2000 images and thresholding. 20 classes changed to 30 classes
# version 4.01 = same as 4.0 except 200 images, batch size 50. for testing purposes. not for Autoencoders. img size (64,128)
# version 4.1 = same as 4.0 except 2500 images
# version 4.2 = same as 4.0 except 1000 images 
# version 4.21 = same as 4.21 except 1000 images 128 batch size 
# version 4.3 = to test different models and avg. same as 4.0 except 500 images. (old 4.3 idk what it was)
# version 5.0 = same as 3.5 except maxN = 30000, no thresholding, no images/class, loss_fc = FocalLoss
# version 5.1 = same as 5.0 except maxN = 56000 which is similar N to 4.1 (56111)
# version 5.2 = same as 5.0 except maxN = 100000
# version 5.3 = same as 4.1 except with focal loss. to test against threshold w/ focal loss. 
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
# version 10.0 = same as 4.0 except batch_size = 50, 100 images and train_AE = True (training the autoencoder). so transforms is with rescale to (128, 256)
# version 10.1 = same as 10.0 except batch_Size = 256, 1000 images, EarlyStopping(patience=20, mode='min')
# version 10.2 = same as 10.1 except 800 epochs
# version 11.0 = to test GoogleNet image size. same as 4.0 except 200 images, Rescale(32,64)
# version 11.1 = same as 11.0 except Rescale(64, 128)
# version 11.2 = same as 11.0 except Rescale(128, 256)
# version 11.3 = same as 11.0 except Rescale(256, 256) out of memory so decreasing batch size to 128
# version 11.4 = same as 11.0 except Rescale(224, 224) and normalization. transforms.Normalize(mean=[0.449], std=[0.226]) added after toTensor. batch size = 128
# version 12.0 = to train variational autoencoder. same as 10.0 for testing purposes. loss_function = VAE_Criterion
# version 12.1 = same as 12.0 except batch_size = 256, 1000 images, 800 epochs, split [0.8,0.1,0.1]
# version 12.2 = same as 12.1 except updated loss function
class Hyperparameters:
    version=4.3
    learning_rate = 0.0003
    number_of_epochs = 800
    momentum = 0.9
    optimizer = optim.Adam
    loss_function = VAE_Criterion
    es = EarlyStopping(patience=20)
    batch_size = 256
    scheduler = None
    pp_strategy = "thresholding"
    data_splits = [0.8,0.1,0.1] #normally [0.6, 0.2, 0.2]
    maxN = None 
    minimum = None
    train_AE = False 
    number_of_images_per_class = 500
    transformations = transforms.Compose([Rescale((64, 128)), ToTensor()]) #transforms.Compose([Rescale((224, 224)),ToTensor(), Normalize(mean=[0.449], std=[0.226])]) # GN fancytransforms.Compose([RandomCrop(16), Rescale((64, 128), multiple=True), ToTensor(multiple=True)])

