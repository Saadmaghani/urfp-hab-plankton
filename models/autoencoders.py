from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

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
# version 3.1 = 1 <-> 27 <-> 9 <-> 3 => (16, 32) => 13492
class Simple_AE(nn.Module):
    version = 3.0

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

# script copied form https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
# author: sksq96

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, size = (256, 8, 16)):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        size = self.size
        return input.view(input.size(0), size[0], size[1], size[2])

# version 1.0 = kernal_size=3, stride=1, padding=1. Maxpool after every 1 layers. 4 layers. z_dim = 32, h_dim = 8,16,256
# version 1.1 = z_dim = 64
# version 1.11 = same as 1.1 except for MNIST dataset
# version 1.12 = same as 1.1 except with 2 classes
# version 1.2 = z_dim = 128
# version 1.3 = z_dim = 64, 5 layers. h_dim = 4*8*512
# version 1.4 = z_dim = 64, 3 layers. h_dim = 16*32*128
# version 1.5 = z_dim = 16, h_dim = 256,8,16. 2 classes
# version 1.6 = same as 1.5 except z_dim = 8
# version 2.0 = 1 conv layer, z_dim = 32, h_dim = 32,64,128, 1 class (detritus)
# version 2.1 = 2 conv layer, z_dim = 32. h_dim = 64,32,64
# version 2.2 = 3 conv layer, z_dim = 32. h_dim = 128,16,32
# version 2.3 = 4 conv layer, z_dim = 32, h_dim = 256,8,16
# version 3.0 = 3 conv layer, z_dim = 2, h_dim = 128,16,32
# version 3.1 = 3 conv layer, z_dim = 4
# version 3.2 = 3 conv layer, z_dim = 8
# version 3.3 = 3 conv layer, z_dim = 16
# version 3.4 = 3 conv layer, z_dim = 64
class CNN_VAE(nn.Module):
    version = 3.4

    def __init__(self, image_channels=1, h_dim=8*16*256, z_dim=64):
        super(CNN_VAE, self).__init__()
        if self.version == 1.5:
            z_dim = 16
        elif self.version == 1.0:
            z_dim = 32
        elif self.version==1.1 or self.version == 1.3:
            z_dim=64
        elif self.version==1.2:
            z_dim=128

        z_dim = 64

        self.z_dim = z_dim
        self.h_dims = (128, 16, 32)
        h_dim = self.h_dims[0]*self.h_dims[1]*self.h_dims[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(size=self.h_dims),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        d = self.get_device()
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(d)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


    def get_sample(self):
        d = self.get_device()
        z = torch.randn(1, self.z_dim).to(d)
        sample = self.decode(z)

        return sample

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu") 


# code gotten from https://graviraja.github.io/vanillavae/#
class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class VAE_Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted

# version 1.0 = as gotten from source. input_dim = 128*256, latent_dim = 128
# version 1.1 = same as 1.0 except latent_dim = 256 
# version 2.x = same as 1.x versions except with less classes. in this case 149 boi (chaetoceros flagellete)
class VAE(nn.Module):
    version = 2.0
    
    def __init__(self, input_dim = 128*256, latent_dim = 128):
        super().__init__()
        hidden_dim = 256
        self.latent_dim = latent_dim

        self.enc = VAE_Encoder(input_dim, hidden_dim, latent_dim)
        self.dec = VAE_Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        
        x = x.view(-1, 128*256)

        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var

    def get_sample(self):
        d = self.get_device()
        z = torch.randn(1, self.latent_dim).to(d)
        sample = self.dec(z)

        return sample

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu") 

    def __str__(self): 
        return type(self).__name__ + "_" + str(self.version)
