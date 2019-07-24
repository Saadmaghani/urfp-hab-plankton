import torch

class Trainer:
	def __init__(self, epochs, loss_fn, optimizer, lr = 0.01, momentum=0.9):
		self.epochs = epochs
		self.criterion = loss_fn()
		self.optimizer = optimizer
		self.lr = lr
		self.momentum = momentum

	def train(self, model, trainloader):
		model = model.float()
		optimizer = self.optimizer(model.parameters(), lr=self.lr, momentum=self.momentum)
		loss = None
		for epoch in range(self.epochs):
			running_loss = 0.0

			for i, data in enumerate(trainloader, 0):
				#get the unputs; data is a list of [inputs, labels]
				inputs, labels = data['image'], data['encoded_label'].float()
				
				#zero the param gradients
				optimizer.zero_grad()
				
				#forward + backward + optimize
				outputs = model(inputs.float())
				
				loss = self.criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				#print statistics - have to get more of these
				running_loss += loss.item()
				if i % 2000 == 1999:  # print every 2000 mini-batches
					print('model: ', model,' - [%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss/2000))
					running_loss = 0.0
			self._save_partial_model(model, epoch, loss, optimizer)
		self._save_full_model(model)
		print('Finished Training')

	def _save_partial_model(self, model, epoch, loss, optimizer):
		model_name = type(model).__name__
		version = type(model).Version
		path_to_statedict = './models/'+model_name+'-'+str(version)+'.tar' 
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss
			}, path_to_statedict)

	def load_partial_model(self, path_to_statedict):
		model = TheModelClass(*args, **kwargs)
		optimizer = TheOptimizerClass(*args, **kwargs)

		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']

		model.eval()
		# - or -
		model.train()

	def _save_full_model(self, model):
		# saving model
		model_name = type(model).__name__
		version = type(model).Version
		path_to_statedict = './models/'+model_name+'-'+str(version)+'.pth' 
		torch.save(net.state_dict(), path_to_statedict)

	def load_full_model(self, path_to_statedict):
		model = Net()
		model.load_state_dict(torch.load(path_to_statedict))
		model.eval()

	def test(self, model, testloader): #stats finder
		all_preds = torch.LongTensor()
		all_targets = torch.LongTensor()		
		with torch.no_grad():

			for data in testloader:
				inputs, labels = data['image'], data['encoded_label'].float()
				_, labels = torch.max(labels, 1)

				outputs = data['encoded_label'].float() #model(inputs.float())
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
        self.target = y_true
        self.pred = y_pred

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
        
        