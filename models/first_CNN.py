# so we have the dataset, now the CNN LOL. taken from: simple_CNN
#Define a CNN
# https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/

import torch.nn as nn
import torch.nn.functional as F

class firstCNN(nn.Module):
	Version = 1.0

	def __init__(self):
		super(firstCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 27 * 60, 120) # 27 * 60 from image dimension, see implementation doc for details
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 88) # 87 because some classes have no images

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 27 * 60)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
