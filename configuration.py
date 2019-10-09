import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from training import EarlyStopping, FocalLoss

"""
versions:
1.x - defaults + just images
2.x - adam optimizer
3.x - early stopping
4.x - thresholding
5.x - focal loss + proportional reduction of classes
6.x - focal loss + proportional reduction + minimum 
7.x - focal loss + data augmentation + thresholding. 

strategies (preprocessing):
1.0 - "thresholding" | thresholding
2.0 - "propReduce" | proportional reduction
2.1 - "propReduce_min" | proportional reduction with lower bound 
3.0 - "augmentation" | data augmentation
3.1 - "augmentation_max" | data augmentation with upper bound  

"""

# default:
# lr = 0.01, optim - optim.SGD, epochs = 10, es = None, loss_fun = nn.MSELoss, momentum = 0.9, batch size = 128, es w/ patience=10

# versions correspond to hyperparameters.
# version 1.0 = 500 images
# version 1.1 = 2000 images
# version 1.2 = 1000 images
# version 2.0 = lr - 0.0003, optim - adam, 500 images
# version 3.0 = lr - 0.0003, optim - adam, 500 images, es = Early Stopping w/ patience 10, epochs = 200
# version 3.1 = minibatch_size = 256, else all same with version 3.0
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
# version 6.0 = same as 5.0 except minimum = 100. this minimum means that if there are less than min images, include all st n = min(N, minimum). This will remove the population dist. bias but accuracies might increase. lets see
# version 6.1 = same as 6.0 except minimum = 200
# version 6.2 = same as 6.0 except minimum = 300
# version 6.3 = same as 6.0 except minimum = 400
# version 6.4 = same as 6.0 except minimum = 500
# version 6.5 = same as 6.0 except minimum = 600
# version 7.0 = thresholding + data augmention Test with "augmentation"=T, "thresholding"=T, "number_of_images_per_class"=200 and "minimum"=100. no MaxN, same as 3.0 
# version 7.1 = same as 7.0 except minimum = 400. max = 2500 images.

class Hyperparameters:
    version=7.1
    learning_rate = 0.0003
    number_of_epochs = 200
    momentum = 0.9
    optimizer = optim.Adam
    loss_function = FocalLoss
    es = EarlyStopping(patience=20)
    batch_size = 256
    scheduler = None
    pp_strategy = "augmentation_max"
    maxN = None 
    minimum = 400
    number_of_images_per_class = 2500
