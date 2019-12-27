import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math

class Pyramid(nn.Module):

    def __init__(self, n_sets_of_inputs, in_features, out_features, bias=True):
        super(Pyramid, self).__init__()

        self.n = n_sets_of_inputs

        self.in_features = in_features
        self.out_features = out_features
    
        self.weight = Parameter(torch.Tensor(2*(self.n-1), out_features, in_features))
        #self.weight = [Parameter(torch.Tensor(out_features, in_features)) for x in range(2*(n_sets_of_inputs-1))]
        if bias:
            self.bias = Parameter(torch.Tensor(self.n-1, out_features))
            #self.bias = [Parameter(torch.Tensor(out_features)) for x in range(n_sets_of_inputs-1)]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _linear_mult(self, x1, x2, w1, w2, b):
        out = x1.matmul(w1.t()).add(x2.matmul(w2.t()))
        out += b
        return out

    def forward(self, inputs):

        outputs = []
        for i in range(self.n-1):
            x0 = 0
            x1 = 1
            out = self._linear_mult(inputs[x0], inputs[x1], self.weight[x0*2], self.weight[(x1*2)-1], self.bias[i])
            x0 += 1
            x1 += 1
            outputs.append(F.relu(out))

        return outputs 


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# version 1.0 = 3 models - GoogleNet, vgg19_bn, resnet
class N_Parallel_Models(nn.Module):
    version = 1.0

    def __init__(self, tl_models=[], freeze=None, pretrain=False, autoencoder=None):
        super(N_Parallel_Models, self).__init__()

        tl_models = [models.googlenet, models.resnet50, models.vgg19_bn]
        self.models = [m(pretrained=pretrain) for m in tl_models]

        self.models[0].fc = nn.Linear(1024, 2048) #googlenet
        self.models[1].fc = nn.Linear(2048, 2048) #resnet
        self.models[2].classifier = nn.Sequential(nn.Linear(25088, 2048), nn.ReLU(inplace=True), nn.Dropout(p=0.5, inplace=False))
    
        self.pyramid_layers = nn.Sequential(Pyramid(3, 2048, 2048), Pyramid(2, 2048, 30))

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.repeat(1,3,1,1)
        xs = []
        for model in self.models:
            temp = model(x)
            if not isinstance(temp, torch.Tensor):
                temp = temp.logits
            xs.append(temp)

        xs = self.pyramid_layers(xs)

        return self.softmax(xs[0])

    def __str__(self):
        return type(self).__name__ + "_" + str(self.version)




"""  ABANDONED  """
# version 1.0 = vgg19_bn
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

