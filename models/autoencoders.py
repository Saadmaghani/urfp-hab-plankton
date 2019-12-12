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
# version 2.0 = 3 <-> 09 <-> 3 => 32, 64 => 23337
# version 2.1 = 3 <-> 27 <-> 9 <-> 3  => 16, 32 => 80322
# version 3.0 = 1 <-> 09 <-> 3 => (32, 64) => 8805
# version 3.1 = 1 <-> 27 <-> 9 <-> 3 => 13492
class Simple_AE(nn.Module):
    version = 3.1

    def __init__(self):
        super(Simple_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 27, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(27, 9, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(9, 3, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 9, 2, stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(9, 27, 2, stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(27, 1, 2, stride = 2),
            nn.Sigmoid())

    def forward(self, x):
        if self.version >=2.0 and self.version < 3.0:
            x = x.repeat((1,3,1,1))
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent(self, x):
        if x.shape[1] == 1 and self.version >=2.0 and self.version < 3.0:
            x = x.repeat((1,3,1,1))
        return self.encoder(x)

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

