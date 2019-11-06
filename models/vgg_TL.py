from torchvision import models
import torch.nn as nn
import torch

# version 1.0 = vgg16
# version 1.1 = vgg16_bn
# version 1.2 = vgg16_bn as feature extractor
# version 1.3 = vgg16 as feature extractor
# version 1.4 = vgg11_bn
# version 1.5 = vgg19_bn ***
# version 1.6 = vgg19
# version 1.7 = vgg19_bn with 3 input channels ****
# version 1.8 = vgg19_bn with 3 input channels & output of all 96 classes
# version 1.9 = vgg19_bn, 3 input channels, 20 classes, iteratively find out to how many layers we should randomly initializes
class VGG(nn.Module):
    version = 1.9

    def __init__(self, randomInitLayers = 1, freeze=None, pretrain = True):
        super(VGG, self).__init__()

        self.version = str(self.version) + "." + str(randomInitLayers)

        self.model = models.vgg19_bn(pretrained=pretrain)
        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 20)


        layer = 0
        for p in reversed(list(self.model.parameters())):
            if len(p.shape) == 1:
                nn.init.zeros_(p)
            else:
                nn.init.xavier_uniform_(p) 
            layer += 0.5
            if layer >= randomInitLayers:
                break


        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)


# version 1.0 = wide_resnet101_2 with 3 input channels & 20 outputs
class WideNet(nn.Module):
    version = 1.0

    def __init__(self, freeze=None, pretrain=True):
        super(WideNet, self).__init__()

        self.model = models.wide_resnet101_2(pretrained=pretrain)

        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 20)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

# version 1.0 = alexnet with 3 input channels & 20 outputs
class AlexNet(nn.Module):
    version = 1.0

    def __init__(self, freeze=None, pretrain=True):
        super(AlexNet, self).__init__()

        self.model = models.alexnet(pretrained=pretrain)

        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 20)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)



# version 1.0 = googlenet with 3 input channels & 20 outputs
# version 1.1 = all outputs (94)
# version 1.2 = 30 outputs
# version 2.0 = 16 random crops, 16 outputs, average the outputs = answer
# version 3.0 = 16 random crops as minibatch, reshape into 1 minibatch 16*1024 as input into FC
# version 4.0 = same as 1.2 except with auto encoder
class GoogleNet(nn.Module):
    version = 4.0

    # used with version 3.0
    class ReshapeLayer(nn.Module):
        def __init__(self, tup):
            super(GoogleNet.ReshapeLayer, self).__init__()
            self.reshape_tup = tup

        def forward(self, x):
            x = torch.reshape(x, self.reshape_tup)
            return x

    def __init__(self, freeze=None, pretrain=True):
        super(GoogleNet, self).__init__()

        self.model = models.googlenet(pretrained=pretrain)

        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)3
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.version == 3.0:
            self.model.avgpool = nn.Sequential(self.model.avgpool, GoogleNet.ReshapeLayer((1,-1)))
            self.model.fc = nn.Linear(1024*16, 30)
        else:
            self.model.fc = nn.Linear(1024, 30)

        self.softmax = nn.Softmax()

    def forward(self, x):

        if self.version == 2.0:
            x = x.repeat(1,1,3,1,1)
            sums = None
            for i in range(x.shape[1]):
                xs = x[:,i:i+1].squeeze(1)
                xs = self.model(xs)
                xs = self.softmax(xs)
                if sums is None:
                    sums = xs
                else:
                    sums = torch.add(sums, xs)
            x = torch.div(sums, x.shape[1])
        elif self.version == 3.0:
            x = x.reshape(x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            x = x.repeat(1,3,1,1)
            x = self.model(x)
            x = self.softmax(x)
        elif self.version == 4.0:
            
            x = x.repeat(1, 3, 1, 1)
            x = self.model(x)
            x = self.softmax(x)
        else:
            x = x.repeat(1, 3, 1, 1)
            x = self.model(x)
            x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

# version 1.0 = resnet18 with 20 outputs
# version 1.0 = resnet50 with 20 outputs
class ResNet(nn.Module):
    version = 1.1

    def __init__(self, freeze=None, pretrain=True):
        super(ResNet, self).__init__()

        self.model = models.resnet50(pretrained=pretrain)

        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(2048, 20)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

