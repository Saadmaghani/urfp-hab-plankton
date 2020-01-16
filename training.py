import torch
import time
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import math
import sys

class Trainer:
    def __init__(self, HP_version, epochs, loss_fn, optimizer, scheduler = None, lr = 0.01, momentum=0.9, autoencoder=False):
        self.epochs = epochs
        self.hp_version = HP_version
        self.criterion = loss_fn()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.momentum = momentum

        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        self.autoencoder = autoencoder

        
    def train(self, model, trainLoader, validLoader, earlyStopping = None, partialModelFile = None, save = True):
        print(self.device)
        self.timeStart = time.time()
        all_train_acc = []
        all_valid_acc = []

        model = model.float()
        model.to(self.device)
        if self.optimizer == torch.optim.Adam:
            optimizer = self.optimizer(model.parameters(), lr=self.lr)
        else:
            optimizer = self.optimizer(model.parameters(), lr=self.lr, momentum=self.momentum)

        scheduler = None if self.scheduler is None else self.scheduler['scheduler'](optimizer, self.scheduler['step_size'])

        epoch = 0
        
        if partialModelFile is not None:
            model, optimizer, epoch = load_partial_model(model, optimizer, partialModelFile)
            
        model.to(self.device)
            
        loss = None
        valid_acc = None
        train_acc = None
        other_stats = None

        # version 5.x GoogleNet. other_stats = avg. Confidence 
        if str(model)[:-1] == "GoogleNet_5.":
            other_stats = {"avg_confidence":[]}

        best_model_weights = copy.deepcopy(model.state_dict())
        if self.autoencoder:
            best_acc = sys.float_info.max
        else:
            best_acc = 0.0

        while epoch < self.epochs:
            running_loss = 0.0
            if scheduler is not None:
                    scheduler.step()

            for i, data in enumerate(trainLoader, 0):
                #get the unputs; data is a list of [inputs, labels]
                inputs, labels = data['image'], data['encoded_label'].to(self.device).float()


                if type(inputs) is list:
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].to(self.device).float()
                else:
                     inputs = inputs.to(self.device).float()
                
                #zero the param gradients
                optimizer.zero_grad()

                #forward + backward + optimize
                outputs = model(inputs)

                # version 5.x GoogleNet. save avg confidence
                if str(model)[:-1] == "GoogleNet_5.":
                    _, conf = outputs 
                    if "totalConfs" in vars():
                        totalConfs = torch.cat((totalConfs, conf), 0)
                    else:
                        totalConfs = conf
                # training autoencoder:
                if self.autoencoder:
                    # for normal AE, uncomment the two lines below
                    #if outputs.shape[1] == 3:
                    #    inputs = inputs.repeat(1,3,1,1)
                    loss = self.criterion(outputs, inputs)
                else:
                    loss = self.criterion(outputs, labels)

                loss.sum().backward()
                optimizer.step()
                
                #print statistics - have to get more of these
                running_loss += loss.sum().item()
                #print("batch no.:",i)
                if i % 10 == 0:
                    #every 10 batches print - loss, training acc, validation acc
                    if self.autoencoder:
                        train_loss, _ = self.test_autoencoder(model, trainLoader)
                        valid_loss, _ = self.test_autoencoder(model, validLoader)
                        # log loss taken as its a cumulative loss
                        train_acc = train_loss.log().tolist() #simple ae torch.mean(train_loss).tolist()
                        valid_acc = valid_loss.log().tolist() #simple ae torch.mean(valid_loss).tolist()
                        print('Running Training Loss:', running_loss)
                        print('Training Loss:', train_acc)
                        print('Valid Loss:', valid_acc)
                        if valid_acc < best_acc: 
                            best_acc = valid_acc
                            best_model_weights = copy.deepcopy(model.state_dict())
                    else:
                        train_pred, train_target, _ = self.test(model, trainLoader)
                        valid_pred, valid_target, _ = self.test(model, validLoader)
                        train_acc = accuracy_score(train_target.cpu(), train_pred.cpu())
                        valid_acc = accuracy_score(valid_target.cpu(), valid_pred.cpu())      

                        if str(model)[:-1] == "GoogleNet_5.":
                            print('current avg. confidence:', totalConfs.mean().item())
                            other_stats['avg_confidence'].append(totalConfs.mean().item())
                            del totalConfs

                        print('Running Training Loss:', running_loss)
                        print('Training Accuracy:', train_acc)
                        print('Valid Accuracy:', valid_acc)
                        if valid_acc > best_acc:
                            best_acc = valid_acc
                            best_model_weights = copy.deepcopy(model.state_dict())

                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    running_loss = 0.0

                    all_train_acc.append(train_acc)
                    all_valid_acc.append(valid_acc)
                
            epoch += 1
            if save == True:
                self._save_partial_model(model, epoch, loss, optimizer)

            #early stopping checked every epoch rather than every minibatch
            if earlyStopping is not None and earlyStopping.step(valid_acc):
                break
        
            model.load_state_dict(best_model_weights)
        if save == True:
            self._save_full_model(model)
        print('Finished Training')
        self.timeEnd = time.time()
        return all_train_acc, all_valid_acc, epoch, other_stats
    
        
    def getTime(self):
        if hasattr(self, "timeEnd") and hasattr(self, "timeStart"):
            time_delta = self.timeEnd - self.timeStart
            secs = time_delta % 60
            mins = time_delta // 60
            hours = mins // 60
            mins = mins % 60
            hms = str(hours) + ":" + str(mins) + ":" + str(secs)
        else:
            hms = "xx:xx:xx"
        return hms

     
    def _save_partial_model(self, model, epoch, loss, optimizer):
        path_to_statedict = './models/'+str(model)+"-"+str(self.hp_version)+'.tar' 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path_to_statedict)

    def _save_full_model(self, model):
        # saving model
        path_to_statedict = './models/'+str(model)+"-"+str(self.hp_version)+'.pth' 
        torch.save(model.state_dict(), path_to_statedict)

    
    def test_autoencoder(self, model, testloader):
        if not self.autoencoder:
            print("error. self.autoencoder = ", str(self.autoencoder))
            return
        all_loss = 0 # vae: 0  # simple AE: torch.FloatTensor().to(self.device)  

        all_fnames = []
        model.to(self.device)
        
        with torch.no_grad():

            for data in testloader:
                inputs, _ = data['image'].to(self.device).float(), data['encoded_label'].to(self.device).float()
                #inputs, _ = data[0].to(self.device).float(), data[1].to(self.device).float()
                outputs = model(inputs)

                loss = self.criterion(outputs, inputs)

                all_loss += loss #Simple AE: torch.cat((all_loss, sumsquare.view(1)), 0)
                all_fnames.extend(data['fname'])

        return all_loss, all_fnames

    def test(self, model, testloader):
        all_preds = torch.LongTensor().to(self.device)
        all_targets = torch.LongTensor().to(self.device)

        all_fnames = []
        model.to(self.device)
        
        with torch.no_grad():

            for data in testloader:
                inputs, labels = data['image'].to(self.device).float(), data['encoded_label'].to(self.device).float()
                
                _, labels = torch.max(labels, 1)

                outputs = model(inputs)

                # version 5.x GoogleNet has outputs = (outputs, confidence)
                if str(model)[:-1] == "GoogleNet_5.":
                    outputs, confs = outputs 
                    idxs, _ = torch.nonzero(confs>model.threshold, as_tuple=True)
                    idxs = torch.unique(idxs)
                    outputs = outputs[idxs]
                    labels = labels[idxs]

                _, predicted = torch.max(outputs.data, 1)
                #print("labels:", labels)
                #print("predicted:", predicted)
                #print("~~~~~~~~~~~~~~~~")
                all_preds = torch.cat((all_preds, predicted), 0)
                all_targets = torch.cat((all_targets, labels), 0) 
                
                #if total >=10:
                #   break
                
                all_fnames.extend(data['fname'])

        return all_preds, all_targets, all_fnames

# script copied from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
# author: Stefano Nardo, Github: stefanonardo
class EarlyStopping(object):
    def __init__(self, mode='max', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def load_partial_model(self, model, optimizer, path_to_statedict):
    checkpoint = torch.load(path_to_statedict)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    #model.eval()
    # - or -
    model.train()

    return model, optimizer, epoch


def load_partial_model_eval(model, path_to_statedict):
    checkpoint = torch.load(path_to_statedict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # - or -
    #model.train()
    return model


def load_full_model(model, path_to_statedict):
    state_dict = torch.load(path_to_statedict, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()

    del state_dict

    return model



import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# version 1.0 = classifierLoss = BCE Loss, lambda = 1
class ConfidenceLoss(nn.Module):
    version=1.0

    def __init__(self, classifierLoss = nn.BCELoss, lambda_normalizer=1):
        super(ConfidenceLoss, self).__init__()
        self.classifierLoss = classifierLoss()
        self.lambda_normalizer = lambda_normalizer

    def forward(self, output_from_model, input_to_model):
        softmax_classes, sigmoid_confidence = output_from_model
        classifier_loss = self.classifierLoss(softmax_classes, input_to_model)* self.lambda_normalizer
        loss = (sigmoid_confidence**2) * (classifier_loss**2) + (1-sigmoid_confidence)**2

        #loss = loss.repeat(1,2)

        return loss


# script copied from https://graviraja.github.io/vanillavae/#
# author: graviraja
class VAE_Criterion(nn.Module):
    def __init__(self):
        super(VAE_Criterion, self).__init__()

    def forward(self, output_from_model, input_to_model):
        # forward pass
        x_sample, z_mu, z_var = output_from_model

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_sample, input_to_model, size_average=False)

        # kl divergence loss
        #kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)  # error in Loss function
        kl_loss = 0.5 * torch.sum(-torch.log(z_var) + z_mu**2 - 1.0 - z_var)

        # total loss
        loss = recon_loss + kl_loss
        return loss


# script copied form https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
# author: sksq96
class CNNVAE_Criterion(nn.Module):
    def __init__(self):
        super(CNNVAE_Criterion, self).__init__()

    def forward(self, output_from_model, x):
        recon_x, mu, logvar = output_from_model

        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        """
        print("#####################################")
        print("BCE+KLD:", (BCE + KLD).item())
        print("KLD:", KLD.item())
        print("BCE:", BCE.item())
        """
        return BCE + KLD


# script copied from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# author: Allen Qin, Github: qinfubiao
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
