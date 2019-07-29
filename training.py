import torch
import time

class Trainer:
	def __init__(self, HP_version, epochs, loss_fn, optimizer, lr = 0.01, momentum=0.9, useCuda = False):
		self.epochs = epochs
		self.hp_version = HP_version
		self.criterion = loss_fn()
		self.optimizer = optimizer
		self.lr = lr
		self.momentum = momentum
		self.device = torch.device("cpu")
		if useCuda:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cpu")
        
        
	def train(self, model, trainLoader, validLoader, partialModelFile = None):
		print(self.device)
		self.timeStart = time.time()
		all_train_acc = []
		all_valid_acc = []
		
		
		model = model.float()
		model.to(self.device)
		optimizer = self.optimizer(model.parameters(), lr=self.lr, momentum=self.momentum)
		epoch = 0
        
		if partialModelFile is not None:
			model, optimizer, epoch = self.load_partial_model(model, optimizer, partialModelFile)
            
		model.to(self.device)
		loss = None
		while epoch < self.epochs:
			running_loss = 0.0
            
			for i, data in enumerate(trainLoader, 0):
				#get the unputs; data is a list of [inputs, labels]
				inputs, labels = data['image'].to(self.device), data['encoded_label'].to(self.device).float()

                
				#zero the param gradients
				optimizer.zero_grad()
				
				#forward + backward + optimize
				outputs = model(inputs.float())
				
				loss = self.criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				#print statistics - have to get more of these
				running_loss += loss.item()
				
				
				#every batch print - loss, training acc, validation acc
				train_pred, train_target = self.test(model, trainLoader)
				valid_pred, valid_target = self.test(model, validLoader)
				train_acc = accuracy_score(train_target.cpu(), train_pred.cpu())
				valid_acc = accuracy_score(valid_target.cpu(), valid_pred.cpu())
				
				print('Training Loss:', running_loss)
				print('Training Accuracy:', train_acc)
				print('Valid Accuracy:', valid_acc)
				print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				running_loss = 0.0
				
				all_train_acc.append(train_acc)
				all_valid_acc.append(valid_acc)
		
			epoch += 1
			
			
			self._save_partial_model(model, epoch, loss, optimizer)
		self._save_full_model(model)
		print('Finished Training')
		self.timeEnd = time.time()
		return all_train_acc, all_valid_acc
	
        
	def getTime(self):
		return self.timeEnd - self.timeStart

        
        
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
        
		# model.eval()
		# - or -
		model.train()
        
		return model, optimizer, epoch

	def _save_full_model(self, model):
		# saving model
		path_to_statedict = './models/'+str(model)+"-"+str(self.hp_version)+'.pth' 
		torch.save(model.state_dict(), path_to_statedict)

	def load_full_model(self, model, path_to_statedict):
		model.load_state_dict(torch.load(path_to_statedict))
		model.eval()
		return model

	def test(self, model, testloader): #stats finder
		all_preds = torch.LongTensor().to(self.device)
		all_targets = torch.LongTensor().to(self.device)
		model.to(self.device)
		
		with torch.no_grad():

			for data in testloader:
				inputs, labels = data['image'].to(self.device), data['encoded_label'].to(self.device).float()
				_, labels = torch.max(labels, 1)

				outputs = model(inputs.float())
				_, predicted = torch.max(outputs.data, 1)
				#print("labels:", labels)
				#print("predicted:", predicted)
				#print("~~~~~~~~~~~~~~~~")
				all_preds = torch.cat((all_preds, predicted), 0)
				all_targets = torch.cat((all_targets, labels), 0) 
				#if total >=10:
				#	break

		return all_preds, all_targets


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

class Metrics:
	
    def __init__(self, y_true, y_pred):
        self.target = y_true.cpu()
        self.pred = y_pred.cpu()

    def sample(self, n):
    	random_idx = np.random.choice(list(range(len(self.target))), size = n, replace = False)
    	target = np.array(self.target)[random_idx]
    	pred = np.array(self.pred)[random_idx]
    	return (target, pred)

    def accuracy(self):
        x = accuracy_score(self.target, self.pred)
        print(x)
        return x
    
    def recall(self):
        x = recall_score(self.target, self.pred, average='micro')
        print(x)
        return x
    
    def f_score(self):
        x = f1_score(self.target, self.pred, average='micro')
        print(x)
        return x

    def class_accuracies(self):
        pass
    
    def plot_CM(self, normalize = True):
        cm = confusion_matrix(self.target.view(-1), self.pred.view(-1))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Confusion Matrix - normalized"
        else:
            title = "Confusion Matrix - unnormalized"
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        cax = ax.matshow(cm, cmap = plt.cm.Blues)
        fig.colorbar(cax)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        
