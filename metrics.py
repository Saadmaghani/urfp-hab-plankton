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

        self.target = np.array(self.target)
        self.pred = np.array(self.pred)

    def sample(self, n, classname = None, preprocessor=None, fname=None, working_indices=None, probs=None):
        if working_indices is None:
            working_indices = np.arange(len(self.target))
        if classname is not None:
            if isinstance(classname, str):
                if preprocessor is None:
                    raise TypeError("preprocessor must be given with classname type str")
                else:
                    classname = preprocessor.label_to_onehotInd(classname)
            working_indices = working_indices[np.where(self.target[working_indices] == classname)[0]]

        if n > len(working_indices):
            n = len(working_indices)

        random_idx = np.random.choice(working_indices, size = n, replace = False)
        target = self.target[random_idx]
        pred = self.pred[random_idx]

        if probs is not None:
            probs = np.array(probs)
            probs = probs[random_idx]
            final_probs = [] 
            for i in range(len(pred)):
                trgt_prob, pred_prob = probs[i][target[i]], probs[i][pred[i]]
                final_probs.append('%.4f'%(trgt_prob))
                final_probs.append('%.4f'%(pred_prob))

        if fname is not None:
            fname = np.array(fname)
            imgs = fname[random_idx]
            i=0
            while i < len(pred):
                indxs = np.where(self.target==pred[i])[0]
                pred_img = fname[np.random.choice(indxs, size = 1)]
                imgs = np.insert(imgs, i*2 + 1, pred_img)
                i += 1
            if probs is not None:
                show_plankton(imgs, final_probs)
            else:
                show_plankton(imgs)

        if probs is not None:
            return (pred, target, final_probs)
        else:
            return (pred, target)


        
    def sample_diff(self, n, classname=None, preprocessor=None, fname=None, probs=None):
        wi = np.where(self.target != self.pred)[0]
        return self.sample(n, classname=classname, preprocessor=preprocessor, fname=fname, working_indices=wi, probs=probs)

    def sample_same(self, n, classname=None, preprocessor=None, fname=None, probs=None):
        wi = np.where(self.target == self.pred)[0]
        return self.sample(n, classname=classname, preprocessor=preprocessor, fname=fname, working_indices=wi, probs=probs)



    def accuracy(self):
        return accuracy_score(self.target, self.pred)
    
    def recall(self):
        return recall_score(self.target, self.pred, average='micro')
    
    def f_score(self):
        return f1_score(self.target, self.pred, average='micro')

    def class_accuracies(self, preprocessor = None):

        if preprocessor is None:
            ca_dict = classification_report(self.target, self.pred, output_dict=True)
            return ca_dict
        else:
            class_names = self.get_classnames(preprocessor)
            ca_dict = classification_report(self.target, self.pred, target_names=class_names, output_dict=True)
            return ca_dict
    
    def plot_CM(self, preprocessor = None, normalize = True, diff = False, y=1, title=None, plot_along=None):
        if diff:
            wi = np.where(self.target != self.pred)[0]
        else:
            wi = np.arange(len(self.target))
        
        cm = confusion_matrix(self.target[wi], self.pred[wi])
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Confusion Matrix - normalized" if title is None else title
        else:
            title = "Confusion Matrix - unnormalized" if title is None else title
            
        fig = plt.figure()
        ax = fig.add_subplot(111) 
        if plot_along is not None:
            ax2 = fig.add_subplot(112)
            ax2.plot(plot_along)

        cax = ax.matshow(cm, cmap = plt.cm.Blues)
            
        labels = None
        if preprocessor is not None:
            labels = self.get_classnames(preprocessor)

        fig.set_size_inches(10, 10)
        fig.colorbar(cax)
        plt.title(title, loc='left', y=y)
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
        self.total = 0
        for fn in list_of_files:
            cl = re.search("\/.*\/\d+\/(.*)\/", fn).group(1)
            self.total += 1
            if cl in self.d:
                self.d[cl]['count'] += 1
                self.d[cl]['files'].append(fn)
            else:
                self.d[cl] = {'count': 1, 'files':[]}

    def get_counts(self):
        for key in self.d:
            print("class", key, ": count =",self.d[key]['count'])

    def get_total_count(self):
        return self.total

    def sample(self, n, name=None):
        if name is None:
            name = random.choice(self.d.keys())
        fname = random.sample(self.d[name]['files'], n)
        show_plankton(fname) 

    def plot_counts(self, size=None):
        x = [names for names in self.d.keys()]
        x.sort()
        y = [self.d[names]['count'] for names in x]
        plt.figure(figsize= (15,6) if size is None else size)
        plt.xticks(rotation=90)
        plt.bar(x,y)
        plt.show()

def show_plankton(fnames, probs = None):
    c=0
    for i in range(len(fnames)):
        img = io.imread(fnames[i])
        name = fnames[i].split("/")[3]
        if probs is not None:
            name += " P: "+str(probs[i])
        if c % 2==0:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(img)
            ax1.set_title(name)
        else:
            ax2.imshow(img)
            ax2.set_title(name)
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
