import torch
import time
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import math

class Trainer:
    def __init__(self, HP_version, epochs, loss_fn, optimizer, scheduler = None, lr = 0.01, momentum=0.9, useCuda = False, autoencoder=False):
        self.epochs = epochs
        self.hp_version = HP_version
        self.criterion = loss_fn()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.momentum = momentum
        self.device = torch.device("cpu")
        if useCuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
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
            model, optimizer, epoch = self.load_partial_model(model, optimizer, partialModelFile)
            
        model.to(self.device)
            
        loss = None
        valid_acc = None
        train_acc = None

        best_model_weights = copy.deepcopy(model.state_dict())
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
                
                if self.autoencoder:
                # training autoencoder:
                    loss = self.criterion(outputs, inputs)
                else:
                    loss = self.criterion(outputs, labels)


                loss.backward()
                optimizer.step()
                
                #print statistics - have to get more of these
                running_loss += loss.item()
                
                if i % 10 == 0:
                    #every 10 batches print - loss, training acc, validation acc
                    if self.autoencoder:
                        train_sumSquares, _ = self.test_autoencoder(model, trainLoader)
                        valid_sumSquares, _ = self.test_autoencoder(model, validLoader)
                        train_acc = torch.mean(train_sumSquares).tolist()
                        valid_acc = torch.mean(valid_sumSquares).tolist()
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
        return all_train_acc, all_valid_acc, epoch
    
        
    def getTime(self):
        time_delta = self.timeEnd - self.timeStart
        secs = time_delta % 60
        mins = time_delta // 60
        hours = mins // 60
        mins = mins % 60
        hms = str(hours) + ":" + str(mins) + ":" + str(secs)
        return hms


        
    def _save_partial_model(self, model, epoch, loss, optimizer):
        path_to_statedict = './models/'+str(model)+"-"+str(self.hp_version)+'.tar' 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path_to_statedict)

    def load_partial_model(self, model, optimizer, path_to_statedict):
        checkpoint = torch.load(path_to_statedict)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        #model.eval()
        # - or -
        model.train()
        
        return model, optimizer, epoch
    
    def load_partial_model(self, model, path_to_statedict):
        checkpoint = torch.load(path_to_statedict)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # - or -
        #model.train()
        return model
    
    

    def _save_full_model(self, model):
        # saving model
        path_to_statedict = './models/'+str(model)+"-"+str(self.hp_version)+'.pth' 
        torch.save(model.state_dict(), path_to_statedict)

    def load_full_model(self, model, path_to_statedict):
        model.load_state_dict(torch.load(path_to_statedict))
        model.eval()
        return model

    def test_autoencoder(self, model, testloader):
        if not self.autoencoder:
            return
        all_sumSquares = torch.FloatTensor().to(self.device)  

        all_fnames = []
        model.to(self.device)
        
        with torch.no_grad():

            for data in testloader:
                inputs, _ = data['image'].to(self.device).float(), data['encoded_label'].to(self.device).float()
                outputs = model(inputs)
                sumsquare = torch.sum((outputs - inputs)**2)
                all_sumSquares = torch.cat((all_sumSquares, sumsquare.view(1)), 0)
                all_fnames.extend(data['fname'])

        return all_sumSquares, all_fnames

    def test(self, model, testloader): #stats finder
        all_preds = torch.LongTensor().to(self.device)
        all_targets = torch.LongTensor().to(self.device)

        all_fnames = []
        model.to(self.device)
        
        with torch.no_grad():

            for data in testloader:
                inputs, labels = data['image'].to(self.device).float(), data['encoded_label'].to(self.device).float()
                
                _, labels = torch.max(labels, 1)

                outputs = model(inputs)
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



# script copied from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# author: Allen Qin, Github: qinfubiao
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
