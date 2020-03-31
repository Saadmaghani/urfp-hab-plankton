import os
import pandas as pd
import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFile
from math import sqrt


class PlanktonDataset(Dataset):
    
    def __init__(self, ids, labels, encoded_labels, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_ids = ids
        self.file_labels = labels
        self.encoded_labels = torch.FloatTensor(encoded_labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_name = self.file_ids[index]
        splits = file_name.split("_")
        year = splits[1]


        label = self.file_labels[index] 
        encoded_label = self.encoded_labels[index]
        
        aumgents = None
        if len(splits) == 6:
            augments = splits[5] 
            file_name = "_".join(splits[:5])
        
        img_name = os.path.join(self.root_dir, year, label, file_name)
        img = io.imread(img_name)

        if aumgents is not None:
            if augments == 0:
                img = np.fliplr(img)
            elif augments == 1:
                img = np.flipud(img)
            elif augments == 2:
                img = np.flipud(np.fliplr(img))
            elif augments == 3:
                img = np.rot90(img, 3)
            elif augments == 4:
                img = np.rot90(img)

        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['encoded_label'] = encoded_label
        sample['fname'] = img_name
        
        return sample

# image transformations
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, multiple=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.multi = multiple

    def __call__(self, sample):
        image = sample['image']
        
        if not self.multi:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

            return {'image':img, 'label':sample['label']}
        else:
            h, w = image[0].shape[:2]
            images = []
            no_images = len(image)
            for i in range(no_images):
                if isinstance(self.output_size, int):
                    if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size

                new_h, new_w = int(new_h), int(new_w)

                img = transform.resize(image[0], (new_h, new_w))
                images.append(img)

            return {'image':images, 'label':sample['label']}



# data augmentation techniques
class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    """

    def __init__(self, no_outputs):
        assert sqrt(no_outputs).is_integer()
        self.no_outputs = no_outputs

    def __call__(self, sample):
        image = sample['image']
        
        h, w = image.shape[:2]
        sqrt_OS = sqrt(self.no_outputs)

        new_h, new_w = int(h/sqrt_OS), int(w/sqrt_OS)

        images = []

        for i in range(self.no_outputs):
            top = np.random.randint(0, h - new_h) 
            left = np.random.randint(0, w - new_w)

            crop_image = image[top: top + new_h, left: left + new_w]
            images.append(crop_image)

        return {'image':images, 'label':sample['label']}


# to convert numpy images to torch images
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, multiple=False):
        self.multi = multiple

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if not self.multi:
            image = image.reshape((1,image.shape[0], image.shape[1]))
            return {'image':torch.from_numpy(image), 'label':sample['label']}
        else:
            no_images = len(image)
            images = torch.from_numpy(image[0].reshape((1, 1, image[0].shape[0], image[0].shape[1])))
            for i in range(1, no_images):
                img = torch.from_numpy(image[i].reshape((1, 1, image[i].shape[0], image[i].shape[1])))
                images = torch.cat((images, img), 0)
            return {'image':images, 'label':sample['label']}


# performs img[channel] = (img[channel] - mean[channel])/std[channel]
class Normalize(object):
    # mean: list of mean values channels have to be normalized by
    # std: list of std values channels have to be normalized by
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        for i in range(len(self.mean)):
            img[i] = (img[i]-self.mean[i])/self.std[i]

        return {'image': img, 'label':sample['label']}


class Preprocessor:
    DATA_FOLDER = "./data"

    def __init__(self, years, transformations=None, include_classes=None, strategy=None, maxN=None, train_eg_per_class=None, minimum=None):

        self.seed = 3
        self.years = years
        self.include_classes = include_classes
        self.fnames, self.labels = self._get_lbls_fnames()
        print(len(self.fnames)) 

        if strategy is not None:
            if strategy == "thresholding" and train_eg_per_class is not None: # just upper thresholding
                self.fnames, self.labels = self._threshold_classes(train_eg_per_class)

            elif strategy == "propReduce" and maxN is not None: # just proportional reduction
                self.fnames, self.labels = self._proportinal_reduce_classes(maxN)

            elif strategy == "propReduce_min" and maxN is not None and minimum is not None: # proportional reduction with minimum
                self.fnames, self.labels = self._proportinal_reduce_classes(maxN, minimum)

            elif strategy == "augmentation" and minimum is not None: # augment small classes. no upper limit
                self.fnames, self.labels = self._augment_small_classes(minimum)

            elif strategy == "augmentation_max" and minimum is not None and train_eg_per_class is not None: # augment small classes, upper limit
                self.fnames, self.labels = self._augment_small_classes(minimum, train_eg_per_class)

        self.encoded_labels = self._oneHotEncoding().tolist()
        self.transformations = transforms.Compose([Rescale((64, 128)), ToTensor()]) if transformations is None else transformations
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print(len(self.fnames))


    def create_datasets(self, splits):
        partition, labels, onehot_labels = self._split(splits)
        self.train_dataset = PlanktonDataset(partition['train'], labels['train'], onehot_labels['train'],
            Preprocessor.DATA_FOLDER, transform=self.transformations)

        self.validation_dataset = PlanktonDataset(partition['validation'], labels['validation'], onehot_labels['validation'],
            Preprocessor.DATA_FOLDER, transform=self.transformations)

        self.test_dataset = PlanktonDataset(partition['test'], labels['test'], onehot_labels['test'],
            Preprocessor.DATA_FOLDER, transform=self.transformations)

    # shuffle=False so that the data is trained/validated/tested in exactly the same manner in each run
    def get_loaders(self, lType, batch_size):
        loader = None
        if lType == "train":
            loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
        elif lType == "validation":
            loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
        elif lType == "test":
            loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
        else:
            print("no such dataset loader")
        return loader


    def onehotInd_to_label(self, onehot_ind):
        onehot = [0 for x in range(len(self.encoded_labels[0]))]
        onehot[onehot_ind] = 1
        ind = self.encoded_labels.index(onehot)
        return self.labels[ind]


    def label_to_onehotInd(self, label):
        ind = self.labels.index(label)
        onehot = self.encoded_labels[ind]
        return onehot.index(max(onehot))




    def confident_imgs(self, fnames, batch_size, transformations=None):
        transformations = transforms.Compose([Rescale((64, 128)), ToTensor()]) if transformations is None else transformations
        
        labels = [x.split('/')[3] for x in fnames]

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.labels)
        n = np.max(integer_encoded)
        label_onehot = torch.nn.functional.one_hot(torch.from_numpy(integer_encoded), int(n)+1).tolist()


        fnames = [x.split('/')[-1] for x in fnames]

        dataset = PlanktonDataset(fnames, labels, label_onehot, Preprocessor.DATA_FOLDER, transform=transformations)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return loader


    def _augment_small_classes(self, minimum, maximum = None):
        new_labels = []
        new_fnames = []

        if self.include_classes is not None:
            for class_name in self.include_classes:
                class_idx = np.where(np.array(self.labels) == class_name)
                class_len = len(class_idx[0])

                prev_len0 = len(new_fnames)
                prev_len1 = len(new_labels)

                augmented_fnames = []
                if class_len <= minimum:
                    for i in range(minimum - class_len):
                        np.random.seed(self.seed)
                        rand_idx = np.random.choice(class_idx[0], size = 1)
                        np.random.seed(self.seed) # this will give one and only one data augmentation
                        augment_fname = np.array(self.fnames)[rand_idx][0] +"_"+ str(np.random.randint(0,5)) # one of: y flip, x flip, x-y flip, 90 rotation, 270 rotation
                        augmented_fnames.append(augment_fname)
                    new_fnames.extend(np.array(self.fnames)[class_idx[0]])
                    new_fnames.extend(augmented_fnames)
                    new_labels.extend([class_name]* (len(augmented_fnames) + class_len))

                if maximum is not None:
                    if class_len >= maximum: 
                        np.random.seed(self.seed)
                        rand_idx = np.random.choice(class_idx[0], size = maximum, replace=False)
                        new_fnames.extend(np.array(self.fnames)[rand_idx])
                        new_labels.extend([class_name]* maximum)
                    else:
                        new_fnames.extend(np.array(self.fnames)[class_idx[0]])
                        new_labels.extend([class_name]* class_len)
                else:
                    new_fnames.extend(np.array(self.fnames)[class_idx[0]])
                    new_labels.extend([class_name]* class_len)

                print(class_name,"-",len(new_fnames) - prev_len0, len(new_labels)- prev_len1)
        return new_fnames, new_labels


    # keeps class %s same but reduces total images
    def _proportinal_reduce_classes(self, maxN, minimum = None):
        new_labels = [] 
        new_fnames = []
        n = len(self.labels)
        for class_name in self.include_classes:
            class_idx = np.where(np.array(self.labels) == class_name)
            expected = (len(class_idx[0])*maxN)//n
            if minimum is not None and expected < minimum:
                expected = min(len(class_idx[0]), minimum)
            np.random.seed(self.seed)
            random_idx = np.random.choice(class_idx[0], expected, replace=False)
            image_files = list(np.array(self.fnames)[random_idx])
            new_fnames.extend(image_files)
            new_labels.extend([class_name]*expected)
        return new_fnames, new_labels


    # if #images of class > N, then only randomly select N. else get all
    def _threshold_classes(self, data_per_class):
        new_labels = []
        new_fnames = []
        og_dpc = data_per_class
        if self.include_classes is not None:
            for class_name in self.include_classes:
                data_per_class = og_dpc
                class_idx = np.where(np.array(self.labels) == class_name)
                if len(class_idx[0]) <= data_per_class:
                    #print(class_idx[0])
                    random_idx = class_idx[0]
                    data_per_class = len(class_idx[0])
                else:
                    np.random.seed(self.seed)
                    random_idx = np.random.choice(class_idx[0], size = data_per_class, replace=False)
                #print(class_name, len(random_idx))
                image_files = list(np.array(self.fnames)[random_idx])
                new_fnames.extend(image_files)
                new_labels.extend([class_name]*data_per_class)
        return new_fnames, new_labels


    def _oneHotEncoding(self):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.labels)
        n = np.max(integer_encoded)
        return torch.nn.functional.one_hot(torch.from_numpy(integer_encoded), int(n)+1)

    
    # split into train,test or train,val,test
    def _split(self, pc_splits):
        xTrain, xTest, yTrain, yTest, eyTrain, eyTest = train_test_split(self.fnames, self.labels, self.encoded_labels, test_size = pc_splits[-1], random_state=self.seed)
        xVal = [] 
        yVal = []
        eyVal = []
        if len(pc_splits) == 3:
            xTrain, xVal, yTrain, yVal, eyTrain, eyVal = train_test_split(xTrain, yTrain, eyTrain, test_size = pc_splits[1]/(pc_splits[0]+pc_splits[1]), random_state=self.seed)

        partition = {'train': xTrain, 'validation': xVal, 'test': xTest}
        labels = {'train': yTrain, 'validation': yVal, 'test': yTest}
        encoded = {'train': eyTrain, 'validation': eyVal, 'test': eyTest}
        return (partition, labels, encoded)


    # get all labels and file names
    def _get_lbls_fnames(self):
        fnames = []
        labels = []
        for year in self.years:
            year_path = Preprocessor.DATA_FOLDER+"/"+year
            if os.path.isdir(year_path):
                for class_name in os.listdir(year_path):
                    if self.include_classes is not None and class_name not in self.include_classes:
                        continue
                    c_path = year_path + "/"+class_name

                    if os.path.isdir(c_path):
                        image_files = [x for x in os.listdir(c_path) if ".png" in x]
                        fnames.extend(image_files)
                        labels.extend([class_name]*len(image_files))

        return fnames, labels

