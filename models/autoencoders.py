from torchvision import models
import torch.nn as nn
import torch


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

# version 1.0 = 1 convolutional layer
class Simple_AE(nn.Module):
    version = 1.0

    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            PrintLayer(),
            nn.ReLU(True),
            PrintLayer(),
            nn.MaxPool2d(2),
            PrintLayer())

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(6,3,kernel_size=5),
            PrintLayer(),
            nn.ReLU(True),
            PrintLayer(),
            nn.Sigmoid(),
            PrintLayer())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def latent_representation(self, x):
        return self.encoder(x)


    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

