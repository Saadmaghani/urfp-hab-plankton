from torchvision import models
import torch.nn as nn
import torch


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

# version 1.0 = 2 convolutional layer
class Simple_AE(nn.Module):
    version = 1.0

    def __init__(self):
        super(Simple_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 8, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(8, 32, 2, stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 2, stride = 2),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent(self, x):
        return self.encoder(x)

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

