from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import json


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
        
    def plot_series(series):
    	x = np.arrange(len(series['train']))
    	legend = []
    	for key, value in series.items():
    		plt.plot(x, loss)
    		legend.append(key)
		plt.legend(legend, loc='upper left')
    	plt.show()


def load_json_from_file(filename):
    f = open(filename, "r")
    content = f.read()
    return json.loads(content)

