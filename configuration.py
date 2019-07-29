import torch.optim as optim
import torch.nn as nn

#versions correspond to hyperparameters. 
#version 1.0 = 500 images
#version 1.1 = 2000 images
#version 1.2 = 1000 images
#version 2.0 = lr - 0.0003, optim - adam, 500 images
class Hyperparameters:
	version=2.0
	learning_rate = 0.0003
	number_of_epochs = 10
	momentum = 0.9
	number_of_images_per_class = 500
	optimizer = optim.Adam
	loss_function = nn.MSELoss



