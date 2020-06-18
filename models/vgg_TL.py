from torchvision import models
import torch.nn as nn
import torch


# version 1.0 = googlenet with 3 input channels & 20 outputs
# version 1.1 = all outputs (94)
# version 1.2 = 30 outputs
# version 1.3 = 30 outputs no Chataecores flagellate. instead Mesodinium_sp
# version 1.4 = 31 outputs both Chataecores flagellat and Mesodinium_sp
# version 2.0 = 16 random crops, 16 outputs, average the outputs = answer
# version 3.0 = 16 random crops as minibatch, reshape into 1 minibatch 16*1024 as input into FC
# version 4.0 = same as 1.2 except with auto encoder
# version 5.0 = same as 1.2 except with confidence
# version 5.1 = same as 5.0 except confidence threshold added into training threshold = 0.1
# version 5.11 = same as 5.1 except threshold = 0.2
# version 5.12 = same as 5.1 except threshold = 0.3
# version 5.13 = same as 5.1 except threshold = 0.4
# version 5.14 = same as 5.1 except threshold = 0.5
# version 5.15 = same as 5.1 except threshold = 0.6
# version 5.16 = same as 5.1 except threshold = 0.7
# version 5.17 = same as 5.1 except threshold = 0.8
# version 5.2 = same as 5.1 except training threshold changes to avg. confidence. initial 0.0
# version 5.3 = same as 5.2 except dropping images (during training) with conf < threshold. !! with 13.5 GG
# version 5.4 = same as 5.3 except conf = max x b4 softmax   !! does not work
# version 5.5 = same as 5.3 except conf = max x after softmax !! works pretty good but fairly obvious why. Very high drop rate
# version 5.6 = conf = conf layer after classification b4 softmax!! Does not work
# version 5.7 = Dropout layer b4 conf layer p=0.5
# version 5.71 = Dropout layer b4 conf layer p=0.2
# version 5.72 = Dropout layer b4 conf layer p=0.7
# version 5.8 = same as 5.3 except 94 classes
# version 5.9 = same as 5.3 except conf = difference b/w correct
# version 6.0 = same as 1.2 except for HKUST - 42 classes (newest version)
class GoogleNet(nn.Module):
    version = 6.0

    # used with version 5.0
    class IdentityLayer(nn.Module):
        def __init__(self):
            super(GoogleNet.IdentityLayer, self).__init__()

        def forward(self, x):
            return x

    # used with version 3.0
    class ReshapeLayer(nn.Module):
        def __init__(self, tup):
            super(GoogleNet.ReshapeLayer, self).__init__()
            self.reshape_tup = tup

        def forward(self, x):
            x = torch.reshape(x, self.reshape_tup)
            return x

    def __init__(self, freeze=None, pretrain=True, autoencoder=None, v=None):
        super(GoogleNet, self).__init__()

        self.model = models.googlenet(pretrained=pretrain)
        if v is not None:
            self.version = v

        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.version == 3.0:
            self.model.avgpool = nn.Sequential(self.model.avgpool, GoogleNet.ReshapeLayer((1,-1)))
            self.model.fc = nn.Linear(1024*16, 30)
        elif self.version >= 5.0 and self.version < 6.0:
            self.threshold = 0.0
            self.model.fc = GoogleNet.IdentityLayer()
            self.sigmoid = nn.Sigmoid()
            self.classifier = nn.Linear(1024, 30)
            if self.version == 5.6:
                self.confidence = nn.Linear(30, 1)
            else:
                self.confidence = nn.Linear(1024, 1)
        elif self.version >= 6.0:
            self.model.fc = nn.Linear(1024, 42)
        else:
            self.model.fc = nn.Linear(1024, 30)

        if self.version == 4.0:
            self.autoencoder = autoencoder

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
            with torch.no_grad():
                x = self.autoencoder.get_latent(x)
            x = self.model(x)
            x = self.softmax(x)
        elif self.version >= 5.0 and self.version < 6.0:
            x = x.repeat(1, 3, 1, 1)
            x = self.model(x)
            results = self.softmax(self.classifier(x))
            confidence = self.sigmoid(self.confidence(x))
            x = (results, confidence)
        else:
            x = x.repeat(1, 3, 1, 1)
            x = self.model(x)
            x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)


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
# version 2.0 = vgg19_bn with 3 input channels & 30 classes
class VGG(nn.Module):
    version = 2.0

    def __init__(self, randomInitLayers = 1, freeze=None, pretrain = True):
        super(VGG, self).__init__()

        self.version = str(self.version) #+ "." + str(randomInitLayers)

        self.model = models.vgg19_bn(pretrained=pretrain)
        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 30)

        if self.version == 1.9:
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


# version 1.0 = resnet18 with 20 outputs
# version 1.1 = resnet50 with 20 outputs
# version 2.0 = resnet50 with 30 outputs
class ResNet(nn.Module):
    version = 2.0

    def __init__(self, freeze=None, pretrain=True):
        super(ResNet, self).__init__()

        self.model = models.resnet50(pretrained=pretrain)

        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(2048, 30)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

class ResNeXt(nn.Module):
    version = 1.0

    def __init__(self, freeze=None, pretrain=True, autoencoder=None):
        super(GoogleNet, self).__init__()

        self.model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl', pretrained=pretrain)

        self.model.fc = nn.Linear(1024, 30)
        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)3
        if freeze is not None:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.version == 4.0:
            self.autoencoder = autoencoder
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
            with torch.no_grad():
                x = self.autoencoder.get_latent(x)
            x = self.model(x)
            x = self.softmax(x)
        else:
            x = x.repeat(1, 3, 1, 1)
            x = self.model(x)
            x = self.softmax(x)
        return x

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)


