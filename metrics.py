from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import json


class Metrics:
    
    def __init__(self, y_true, y_pred):

        if not isinstance(y_true, list):
            self.target = y_true.tolist()
        else:
            self.target = y_true
        if not isinstance(y_pred, list):
            self.pred = y_pred.tolist()
        else:
            self.pred = y_pred

    def sample(self, n, fname=None, classname = None, preprocessor = None):

        working_indices = list(range(len(self.target)))
        if classname is not None:
            if isinstance(classname, str) and preprocessor is not None:
                classname = preprocessor.label_to_onehotInd(classname)
            working_indices = np.where(np.array(self.target) == classname)

        random_idx = np.random.choice(working_indices, size = n, replace = False)
        target = np.array(self.target)[random_idx]
        pred = np.array(self.pred)[random_idx]

        if fname is not None:
            imgs = np.array(fname)[random_idx]
            show_plankton(imgs)

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

    def class_accuracies(self, preprocessor = None):

        if preprocessor is None:
            ca_dict = classification_report(self.target, self.pred, output_dict=True)
            return ca_dict
        else:
            uniq_classes = set(self.target)
            class_names = []
            for cl in uniq_classes:
                class_names.append(preprocessor.onehotInd_to_label(cl))
            ca_dict = classification_report(self.target, self.pred, target_names=class_names, output_dict=True)
            return ca_dict
    
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
        x = np.arange(len(series['TrainAcc']))
        legend = []
        for key, value in series.items():
            plt.plot(x, value)
            legend.append(key)
        plt.legend(legend, loc='upper left')
        plt.show()


def show_plankton(fnames):
    pass
    ln = len(fnames)

    fig = plt.figure()
    ax = fig.add_subplot(222)
    plt.imshow(img,cmap = 'gray')
    cax = ax.matshow(cm, cmap = plt.cm.Blues)
    fig.colorbar(cax)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



def load_json_from_file(filename):
    f = open(filename, "r")
    content = f.read()
    return json.loads(content)

