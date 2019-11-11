from torchvision import models
import torch.nn as nn
import torch


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

# version 1.0 = 3 <-> 32 <-> 8
# version 1.1 = 3 <-> 27 <-> 9 
# version 1.2 = 3 <-> 27 <-> 9 <-> 3
# version 1.3 = 1 <-> 27 <-> 9 
# version 1.4 = 1 <-> 27 <-> 9 <-> 1
# version 1.5 = 3 <-> 9 <-> 3
class Simple_AE(nn.Module):
    version = 1.5

    def __init__(self):
        super(Simple_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(9, 3, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2))

        self.decoder = nn.Sequential(          
            nn.ConvTranspose2d(3, 9, 2, stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(9, 1, 2, stride = 2),
            nn.Sigmoid())

    def forward(self, x):
        #x = x.repeat((1,3,1,1))
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent(self, x):
        #if x.shape[1] == 1:
        #    x = x.repeat((1,3,1,1))
        return self.encoder(x)

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

