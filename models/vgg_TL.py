from torchvision import models
import torch.nn as nn

#version 1.0 = vgg16
#version 1.1 = vgg16_bn
#version 1.2 = vgg16_bn as feature extractor
#version 1.3 = vgg16 as feature extractor
#version 1.4 = vgg11_bn
#version 1.5 = vgg19_bn ****
#version 1.6 = vgg19
#version 1.7 = vgg19_bn with 3 input channels
class VGG(nn.Module):
    version = 1.7

    def __init__(self, freeze = False, pretrain = True):
        super(VGG, self).__init__()

        self.model = models.vgg19_bn(pretrained=pretrain)
        #self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 20)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(VGG.version)
