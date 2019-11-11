from torchvision import models
import torch.nn as nn
import cv2

# version 1.0 = vgg19_bn


"""  ABANDONED  """

class TripleArch(nn.Module):
    version = 1.0

    def contrast_increaser(self, x):
        if x > 10:
            x = x*2.5
            if x > 255:
                x = 255
        else:
            x = x*0.4
        return x

    def __init__(self, randomInitLayers = 1, freeze=None, pretrain = True):
        super(TripleArch, self).__init__()
        self.cv2imread = cv2.imread
        self.canny = cv2.Canny # 20, 100


        self.bilateral = cv2.bilateralFilter # 9,75,75 
        self.scharr = cv2.Scharr # -1, 0, 1; -1, 1, 0



        self.version = str(self.version)


        self.model_og = models.vgg19_bn(pretrain=pretrain)
        # self.model.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model_og.parameters():
                param.requires_grad = False

        num_ftrs = self.model_og.classifier[6].in_features
        self.model_og.classifier[6] = nn.Linear(num_ftrs, 20)



        self.model_global = models.vgg19_bn(pretrain=pretrain)
        # self.model_global.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model_global.parameters():
                param.requires_grad = False

        num_ftrs = self.model_global.classifier[6].in_features
        self.model_global.classifier[6] = nn.Linear(num_ftrs, 20)


        self.model_local = models.vgg19_bn(pretrain=pretrain)
        # self.model_local.features[0] = nn.Conv2d(1, 64, 3, padding = 1)
        if freeze is not None:
            for param in self.model_local.parameters():
                param.requires_grad = False

        num_ftrs = self.model_local.classifier[6].in_features
        self.model_local.classifier[6] = nn.Linear(num_ftrs, 20)




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

