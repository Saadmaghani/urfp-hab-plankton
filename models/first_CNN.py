# so we have the dataset, now the CNN LOL. taken from: simple_CNN
#Define a CNN
# https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/

#version 1.1 = all data except mix

import torch.nn as nn
import torch.nn.functional as F

class firstCNN(nn.Module):
	Version = 1.2 

	def __init__(self):
		super(firstCNN, self).__init__()
		self.pool = nn.MaxPool2d(2, 2)
		
		self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
		self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
		
		
		self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
		self.conv4 = nn.Conv2d(64, 64, 3, padding = 1)
		
		self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
		self.conv6 = nn.Conv2d(128, 128, 3, padding = 1)
		
		self.conv7 = nn.Conv2d(128, 256, 3, padding = 1)
		self.conv8 = nn.Conv2d(256, 256, 3, padding = 1)
		
		self.fc1 = nn.Linear(32*256, 128) # 27 * 60 from image dimension, see implementation doc for details
		self.fc2 = nn.Linear(128, 86)
		self.softmax = nn.Softmax() 
		#self.fc3 = nn.Linear(84, 86) # 102 because ignored_classes = ['mix'] and using all data

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool(x)
		
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = self.pool(x)
		
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		x = self.pool(x)
		
		x = x.view(x.size()[0],-1)
		x = F.dropout(x)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.softmax(x)
		return x
