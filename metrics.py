from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import json
from skimage import io, transform


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

    def sample(self, n, classname = None, preprocessor = None, fname=None, working_indices = None):
        if working_indices is None:
            working_indices = np.arange(len(self.target))
        if classname is not None:
            if isinstance(classname, str):
                if preprocessor is None:
                    raise TypeError("preprocessor must be given with classname type str")
                else:
                    classname = preprocessor.label_to_onehotInd(classname)
            working_indices = working_indices[np.where(np.array(self.target)[working_indices] == classname)[0]]

        if n > len(working_indices):
            n = len(working_indices)
        random_idx = np.random.choice(working_indices, size = n, replace = False)
        target = np.array(self.target)[random_idx]
        pred = np.array(self.pred)[random_idx]
        if fname is not None:
            imgs = np.array(fname)[random_idx]
            i=0
            while i < len(pred):
                indxs = np.where(np.array(self.target)==pred[i])[0]
                pred_img = np.array(fname)[np.random.choice(indxs, size = 1)]
                imgs = np.insert(imgs, i*2 + 1, pred_img)
                i += 1
            show_plankton(imgs)

        return (target, pred)


    def sample_diff(self, n, classname=None, preprocessor=None, fname=None):
        working_indices = np.where(np.array(self.target) != np.array(self.pred))[0]
        
        return sample(n, classname, preprocessor, fname, working_indices)


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
            class_names = self.get_classnames(preprocessor)
            ca_dict = classification_report(self.target, self.pred, target_names=class_names, output_dict=True)
            return ca_dict
    
    def plot_CM(self, preprocessor = None, normalize = True):
        if not isinstance(self.target, list):
            cm = confusion_matrix(self.target.view(-1), self.pred.view(-1))
        else:
            cm = confusion_matrix(self.target, self.pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Confusion Matrix - normalized"
        else:
            title = "Confusion Matrix - unnormalized"
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        cax = ax.matshow(cm, cmap = plt.cm.Blues)

        labels = None
        if preprocessor is not None:
            labels = self.get_classnames(preprocessor)

        fig.set_size_inches(10, 10)
        fig.colorbar(cax)
        plt.title(title, loc='left')
        plt.xlabel('Predicted')
        plt.xticks(np.arange(cm.shape[1]), labels, rotation = 'vertical')
        plt.ylabel('True')
        plt.yticks(np.arange(cm.shape[0]), labels)
        plt.show()
        
    def plot_series(series):
        x = np.arange(len(series['TrainAcc']))
        legend = []
        for key, value in series.items():
            plt.plot(x, value)
            legend.append(key)
        plt.legend(legend, loc='upper left')
        plt.show()


    def get_classnames(self, preprocessor = None):
        if preprocessor is not None:
            uniq_classes = set(self.target)
            class_names = []
            for cl in uniq_classes:
                class_names.append(preprocessor.onehotInd_to_label(cl))
            return class_names


def show_plankton(fnames):
    c=0
    for fname in fnames:
        img = io.imread(fname)
        if c % 2==0:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(img)
            ax1.set_title(fname.split("/")[3])
        else:
            ax2.imshow(img)
            ax2.set_title(fname.split("/")[3])
        c += 1
    plt.show()


def load_json_from_file(filename):
    f = open(filename, "r")
    content = f.read()
    return json.loads(content)

