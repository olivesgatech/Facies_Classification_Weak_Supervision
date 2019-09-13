import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from torch.utils import data
import scipy.misc

class patch_loader(data.Dataset): #torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override __len__ and __getitem__
    """
        Data loader for the patch-based deconvnet, seismic volume and all txt lists are loaded when an instance is created
    """
    def __init__(self, split='train', stride=30 ,patch_size=75, is_transform=True,augmentations=None): #initialize the class
        self.root = 'data/' #define the data root directory
        self.split = split #define the split
        self.is_transform = is_transform # transform boolean
        self.augmentations = augmentations #define augmentations, method passed from outside in a variable
        self.n_classes = 7 #define the number of classes
        self.mean = 0.000941 # average of the training data  
        self.patches = collections.defaultdict(list) #means that if a key is not found in the dictionary, then instead of a KeyError being thrown, a new entry is created. The type of this new entry is given by the argument of defaultdict.
        self.patch_size = patch_size #initialize the patch size (length and width of the image)
        self.stride = stride #initialize the stride

        if 'test' not in self.split:  #load seismic volume
            # Normal train/val mode
            self.seismic = self.pad_volume(np.load(pjoin('data','train','train_seismic.npy'))) #load training seismic block and pad it with zeros
            # +1 added to shift the index of the labels to be 1-indexed. This way, ignore_index will be 0.
            self.labels = self.pad_volume(np.load(pjoin('data','train','train_labels.npy'))) + 1 #load training seismic block and pad it with zeros
        elif 'test1' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test1_seismic.npy')) #load test1 seismic data
            self.labels = np.load(pjoin('data','test_once','test1_labels.npy')) + 1 #load test1 labels
        elif 'test2' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test2_seismic.npy')) #load test2 seismic data
            self.labels = np.load(pjoin('data','test_once','test2_labels.npy')) + 1 #load test2 labels
        else:
            ValueError('Unknown split.') #if a new argument in the split then raise unkown split

        if 'test' not in self.split: # Load list.txt of sections/images
            # We are in train/val mode. Most likely the test splits are not saved yet, 
            # so don't attempt to load them.  
            for split in ['train', 'val', 'train_val']: # then for train, val, train_val
                # reading the file names for 'train', 'val', 'trainval'""
                path = pjoin('data', 'splits', 'patch_' + split + '.txt') #create the path to data
                patch_list = tuple(open(path, 'r')) #load the names of the sections
                # patch_list = [id_.rstrip() for id_ in patch_list]
                self.patches[split] = patch_list #put the list of names in a dictionary with a given name
        elif 'test' in split: #if test is in the data
            # We are in test mode. Only read the given split. The other one might not 
            # be available. 
            path = pjoin('data', 'splits', 'patch_' + split + '.txt') #load test list
            file_list = tuple(open(path,'r')) #join the file path
            # patch_list = [id_.rstrip() for id_ in patch_list]
            self.patches[split] = patch_list #store it in the dictionary
        else:
            ValueError('Unknown split.') #if a new argument in the split then raise unkown split

    def pad_volume(self,volume):
        '''
        Only used for train/val!! Not test.
        '''
        assert 'test' not in self.split, 'There should be no padding for test time!'
        return np.pad(volume,pad_width=self.patch_size,mode='constant', constant_values=0) # Pads with zeros
        

    def __len__(self): #method that overrides pytorch methods to get the length of the whole dataset
        return len(self.patches[self.split])

    def __getitem__(self, index): #method that overrides pytorch methods to load the data item given its index in the dataset

        patch_name = self.patches[self.split][index] #call the dictionary and index it
        direction, idx, xdx, ddx = patch_name.split(sep='_') #get the direction, inline index, xline index and depth index ??????
        #ToDO: ask about those lines

        # We padd the volume during train/val by self.patch_size. To account for the
        # shift in the indices, we add self.patch_size **only** if the spit is a train/val split
        shift = (self.patch_size if 'test' not in self.split else 0) #get the correct shift for training and validation
        idx, xdx, ddx = int(idx)+shift, int(xdx)+shift, int(ddx)+shift #get the correct indces after correction

        if direction == 'i': #if direction is inline
            im = self.seismic[idx,xdx:xdx+self.patch_size,ddx:ddx+self.patch_size] #get image from seismic block
            lbl = self.labels[idx,xdx:xdx+self.patch_size,ddx:ddx+self.patch_size] #get labels from seismic block
        elif direction == 'x':    
            im = self.seismic[idx: idx+self.patch_size, xdx, ddx:ddx+self.patch_size] #get image from seismic block
            lbl = self.labels[idx: idx+self.patch_size, xdx, ddx:ddx+self.patch_size] #get labels from seismic block

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl) #perform augmentations
            
        if self.is_transform:
            im, lbl = self.transform(im, lbl) #perfrom transform
        return im, lbl


    def transform(self, img, lbl):
        img -= self.mean #zero mean the image

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T  #transpose the image

        img = np.expand_dims(img,0) #expand dimensions of np array
        lbl = np.expand_dims(lbl,0) #expand dimensions of np array

        img = torch.from_numpy(img)  #convert image to tensor
        img = img.float() #convert tensor to float
        lbl = torch.from_numpy(lbl) #convert label to tensor
        lbl = lbl.long() #convert label to long
                
        return img, lbl #return image and label

    def get_seismic_labels(self):
        # First class in NULL (ignore_index)
        return np.asarray([[0,0,0], [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],[215,48,39]]) #different colors for different classes



    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels() #color map
        r = label_mask.copy() #copy of the labels of the current labels 2D
        g = label_mask.copy() #copy of the labels of the current labels 2D
        b = label_mask.copy() #copy of the labels of the current labels 2D
        for ll in range(0, self.n_classes): #for loop over all classes
            r[label_mask == ll] = label_colours[ll, 0] #all classes with label i get color i for red
            g[label_mask == ll] = label_colours[ll, 1] #all classes with label i get color i for green
            b[label_mask == ll] = label_colours[ll, 2] #all classes with label i get color i for blue
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3)) #concatinate colours
        rgb[:, :, 0] = r / 255.0 #normalize image
        rgb[:, :, 1] = g / 255.0 #normalize image
        rgb[:, :, 2] = b / 255.0 #normalize image
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb #return image

        
class section_loader(data.Dataset):
    """
        Data loader for the section-based deconvnet
    """
    def __init__(self, split='train', is_transform=True,augmentations=None):
        self.root = 'data/'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 7 
        self.mean = 0.000941 # average of the training data  
        self.sections = collections.defaultdict(list)

        if 'test' not in self.split: 
            # Normal train/val mode
            self.seismic = np.load(pjoin('data','train','train_seismic.npy'))
            # +1 added to shift the index of the labels to be 1-indexed. This way, ignore_index will be 0.
            self.labels = np.load(pjoin('data','train','train_labels.npy')) + 1
        elif 'test1' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test1_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test1_labels.npy')) + 1
        elif 'test2' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test2_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test2_labels.npy')) + 1
        else:
            ValueError('Unknown split.')

        if 'test' not in self.split:
            # We are in train/val mode. Most likely the test splits are not saved yet, 
            # so don't attempt to load them.  
            for split in ['train', 'val', 'train_val']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = pjoin('data', 'splits', 'section_' + split + '.txt')
                file_list = tuple(open(path, 'r'))
                file_list = [id_.rstrip() for id_ in file_list]
                self.sections[split] = file_list
        elif 'test' in split:
            # We are in test mode. Only read the given split. The other one might not 
            # be available. 
            path = pjoin('data', 'splits', 'section_' + split + '.txt')
            file_list = tuple(open(path,'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        else:
            ValueError('Unknown split.')


    def __len__(self):
        return len(self.sections[self.split])

    def __getitem__(self, index):

        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')

        if direction == 'i':
            im = self.seismic[int(number),:,:]
            lbl = self.labels[int(number),:,:]
        elif direction == 'x':    
            im = self.seismic[:,int(number),:]
            lbl = self.labels[:,int(number),:]
        
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
            
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()
                
        return img, lbl

    def get_seismic_labels(self):
        return np.asarray([[0,0,0], [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39]])


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
        

class patch_loader_weak(data.Dataset):

    def __init__(self, split='train', patch_size=75, is_transform=True, augmentations=None):
        self.root = 'data/train_weak'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 7
        self.mean = 0.000941 # average of the training data  
        self.patches = collections.defaultdict(list)
        self.patch_size = patch_size

        for split in ['train', 'val', 'trainval']:
            # reading the file names for 'train', 'val', 'trainval'"
            # need to make a file to split dataset to 'train', 'val', 'trainval'"
            path = pjoin(self.root, 'splits', split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [(id_.rstrip(),int(id_.split('_s')[-1])) for id_ in file_list] #change added (   ,int(id_.split('_s')[-1]))
            self.patches[split] = file_list


    def __len__(self):
        return len(self.patches[self.split])

    def __getitem__(self, index):
        img_id = self.patches[self.split][index][0]#change added [0]

        img_id_splits = img_id.split('_')
        img_similarity = int(img_id_splits[-1][1:])/100.0

        im_path = pjoin(self.root, 'images',  img_id + '.mat')
        lbl_path = pjoin(self.root, 'labels','L_' + img_id + '.mat')
        conf_path = pjoin(self.root, 'conf_six_channels','C_' + img_id + '.mat')

        im_data = io.loadmat(im_path)
        im = im_data['img'].astype(np.float64)

        # these labels are 1-indexed
        lbl_data = io.loadmat(lbl_path)
        lbl = lbl_data['classifiedImage'].astype(np.uint8) 

        # Read the image level label:
        img_level_lbl = int(img_id[3]) # 1-indexed
        
        conf_data = io.loadmat(conf_path)
        conf = conf_data['conf'].astype(np.float64)
        conf = conf.reshape((self.patch_size,self.patch_size,6))#added 6 for different channels

        #  NEW STUFF: Try this out:
        layer_below = max(1,img_level_lbl-1)
        layer_above = min(6,img_level_lbl+1)
        mask = (lbl!=layer_below) * (lbl!=img_level_lbl) * (lbl!=layer_above) 
        lbl[mask] = 0 # zero out elements that are not neighboring -- this will pass on the



        if self.augmentations is not None:
            im, lbl, conf = self.augmentations(im, lbl, conf)
            
        if self.is_transform:
            im, lbl, conf = self.transform(im, lbl, conf)
        sim = self.patches[self.split][index][1]  # added line
        return im, lbl, conf , sim #added similarity index

    def transform(self, img, lbl, conf):
        # img = img.astype(np.float64) / 255 # [0,255] => [0,1]
        img -= self.mean 

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)
        conf = np.expand_dims(conf,0)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        conf = torch.from_numpy(conf).float()

        # img=img.view(1,self.patch_size,self.patch_size)
                
        return img, lbl, conf

    def get_seismic_labels(self):
        return np.asarray([[0,0,0], [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39]])


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

