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
        year = file_name.split("_")[1]
        label = self.file_labels[index] 
        encoded_label = self.encoded_labels[index]
        
        img_name = os.path.join(self.root_dir, year, label, file_name)
        img = io.imread(img_name)

        sample = {'image': img, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
            
        sample['encoded_label'] = encoded_label
        sample['image'] = sample['image'].reshape((1,sample['image'].shape[0], sample['image'].shape[1]))

        return sample



# image transformations
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

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


# data augmentation techniques
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h) 
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image':image, 'label':sample['label']}


# to convert numpy images to torch images
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((0, 1))
        return {'image':torch.from_numpy(image), 'label':sample['label']}


class Preprocessor:
    DATA_FOLDER = "./data"

    def __init__(self, years, transformations = None, include_classes = None, thresholding = False, maxN = None, train_eg_per_class = None):
        self.seed = 3
        self.years = years
        self.include_classes = include_classes
        self.fnames, self.labels = self._get_lbls_fnames()
        self.thresholding = thresholding
                
        if maxN is not None and thresholding is False and train_eg_per_class is None:
            self.fnames, self.labels = self._reduce_classes(maxN)
        if train_eg_per_class is not None:
            self.fnames, self.labels = self._normalize_classes(train_eg_per_class)

        self.encoded_labels = self._oneHotEncoding().tolist()
        self.transformations = transforms.Compose([Rescale((64, 128)), ToTensor()]) if transformations is None else transformations
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print(len(self.fnames))


    def _reduce_classes(self, maxN):
        new_labels = []
        new_fnames = []
        n = len(self.labels)
        for class_name in self.include_classes:
            class_idx = np.where(np.array(self.labels) == class_name)
            expected = (len(class_idx[0])*maxN)//n
            np.random.seed(self.seed)
            random_idx = np.random.choice(class_idx[0], expected, replace=False)
            image_files = list(np.array(self.fnames)[random_idx])
            new_fnames.extend(image_files)
            new_labels.extend([class_name]*expected)
        return new_fnames, new_labels


    def _normalize_classes(self, data_per_class):
        new_labels = []
        new_fnames = []
        if self.include_classes is not None:
            for class_name in self.include_classes:
                class_idx = np.where(np.array(self.labels) == class_name)
                np.random.seed(self.seed)
                if self.thresholding is True and len(class_idx[0]) <= data_per_class:
                    random_idx = class_idx[0]
                    data_per_class = len(class_idx[0])
                else:
                    random_idx = np.random.choice(class_idx[0], size = data_per_class, replace=False)
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


    def create_datasets(self, splits):
        partition, labels, onehot_labels = self._split(splits)
        self.train_dataset = PlanktonDataset(partition['train'], labels['train'], onehot_labels['train'],
            Preprocessor.DATA_FOLDER, transform=self.transformations)

        self.validation_dataset = PlanktonDataset(partition['validation'], labels['validation'], onehot_labels['validation'],
            Preprocessor.DATA_FOLDER, transform=self.transformations)

        self.test_dataset = PlanktonDataset(partition['test'], labels['test'], onehot_labels['test'],
            Preprocessor.DATA_FOLDER, transform=self.transformations)


    def onehot_to_label(self, onehot):
        ind = self.encoded_labels.index(onehot)
        return self.labels[ind]


    def get_loaders(self, lType, batch_size):
        loader = None
        if lType == "train":
            loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        elif lType == "validation":
            loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        elif lType == "test":
            loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            print("no such dataset loader")
        return loader

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
