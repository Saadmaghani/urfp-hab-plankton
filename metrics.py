from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import json
from skimage import io, transform
import torchvision
from numpy.polynomial.polynomial import polyfit
import re
import random

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
        wi = np.where(np.array(self.target) != np.array(self.pred))[0]
        
        return self.sample(n, classname=classname, preprocessor=preprocessor, fname=fname, working_indices=wi)


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
        

    def _plot_avg_lines(x, y, size, color):
        q = len(x)//size
        r = len(x)%size
        for i in range(q,0,-1):
            b, m = polyfit(x[(i-1)*size+r : (i*size)+r], y[(i-1)*size+r : (i*size)+r], 1)
            plt.plot(x[(i-1)*size+r : (i*size)+r], b + m * x[(i-1)*size+r : (i*size)+r], 'C'+str(color), linewidth=3)
        b, m = polyfit(x[0:r], y[0:r], 1)
        plt.plot(x[0:r], b + m * x[0:r], 'C'+str(color), linewidth=3)

    def _plot_lobf_deg(x, y, degree, color):
        
        a = polyfit(x, y, degree)
        out = 0
        for i, deg in enumerate(a):
            out += deg*x**i
        plt.plot(x, out, 'C'+str(color), linewidth=3)


    def plot_series(series, title="", Lloc="upper left", Lncol=1, Lbbox=None, show_avgs=None):
        cn = 0

        for key, value in series.items():
            x = np.arange(len(value))

            if show_avgs is not None:
                if isinstance(show_avgs, tuple):
                    avg, show = show_avgs
                    if show:
                        plt.plot(x, value, 'C'+str(cn), label=key)
                    else:
                        plt.plot(0,0, 'C'+str(cn), label=key)
                    Metrics._plot_lobf_deg(x, value, avg, color=cn)

                else:
                    avg = show_avgs
                    plt.plot(x, value, 'C'+str(cn), label=key)
                    Metrics._plot_lobf_deg(x, value, avg, color=cn)

            else:
                plt.plot(x, value, 'C'+str(cn), label=key)

            cn = (cn+1)%10

        if Lbbox is None:
            plt.legend(loc=Lloc, ncol=Lncol)
        else:
            plt.legend(loc=Lloc, ncol=Lncol, bbox_to_anchor=Lbbox)
        plt.title(title)
        plt.show()


    def get_classnames(self, preprocessor = None):
        if preprocessor is not None:
            uniq_classes = set(self.target)
            class_names = []
            for cl in uniq_classes:
                class_names.append(preprocessor.onehotInd_to_label(cl))
            return class_names


class FileHandler:
    def __init__(self, list_of_files):
        self.d = {}
        for fn in list_of_files:
            cl = re.search("\/.*\/\d+\/(.*)\/", fn).group(1)
            if cl in self.d:
                self.d[cl]['count'] += 1
                self.d[cl]['files'].append(fn)
            else:
                self.d[cl] = {'count': 0, 'files':[]}

    def sample(self, n, name=None):
        if name is None:
            name = random.choice(self.d.keys())
        fname = random.sample(self.d[name]['files'], n)
        show_plankton(fname) 

    def plot_counts(self):
        x = [names for names in self.d.keys()]
        x.sort()
        y = [self.d[names]['count'] for names in x]
        plt.xticks(rotation=90)
        plt.bar(x,y)
        plt.show()

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


def show_weights(layer):
    weights = layer.weight.data
    img_grid = torchvision.utils.make_grid(weights)
    img_grid = img_grid.mean(dim=0)
    img_grid = img_grid /2 +0.5
    plt.imshow(img_grid.numpy(), cmap="Greys")
