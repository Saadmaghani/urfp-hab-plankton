import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from training import EarlyStopping

# default:
# lr = 0.01, optim - optim.SGD, epochs = 10, es = None, loss_fun = nn.MSELoss, momentum = 0.9, batch size = 128, es w/ patience=10

#versions correspond to hyperparameters. 
#version 1.0 = 500 images
#version 1.1 = 2000 images
#version 1.2 = 1000 images
#version 2.0 = lr - 0.0003, optim - adam, 500 images
#version 3.0 = lr - 0.0003, optim - adam, 500 images, es = Early Stopping w/ patience 10, epochs = 200
#version 3.1 = minibatch_size = 256, else all same with version 3.0
#version 3.2 = further training of HP 3.1, model 2.2
#version 3.3 = further training of HP 3.1, model 2.2, patience = 20
#version 3.4 = same as 3.1 + lr_scheduler w/ StepLR & step size = 7
#version 3.5 = same as 3.1 except patience = 20
class Hyperparameters:
    version=3.5
    learning_rate = 0.0003
    number_of_epochs = 200
    momentum = 0.9
    number_of_images_per_class = 500
    optimizer = optim.Adam
    loss_function = nn.MSELoss
    es = EarlyStopping(patience=20)
    batch_size = 256
    scheduler = None

