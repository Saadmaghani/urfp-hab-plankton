from torchvision import models
import torch.nn as nn

#version 1.0 = vgg16
class VGG(nn.Module):
	version = 1.0

	def __init__(self, freeze = False, pretrain = True):
		super(firstCNN, self).__init__()

		self.model = models.vgg16(pretrained=pretrain)
		if freeze:
			for param in self.model.parameters():
	    		param.requires_grad = False

		num_ftrs = self.model.classifier[6].in_features
		self.model.classifier[6] = nn.Linear(num_ftrs, 20)

		self.softmax = nn.Softmax()

	def forward(self, x):
		x = self.model(x)
		x = self.softmax(x)

		return x

	def __str__(self):
		return type(self).__name__ + "_"+str(VGG.version)