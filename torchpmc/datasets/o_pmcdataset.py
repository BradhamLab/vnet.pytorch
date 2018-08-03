import numpy as np

import torch
import torch.utils.data as data
from torchpmc import utils
import glob
import os
import os.path
import SimpleITK as sitk
import pandas as pd
from skimage.draw import polygon
from PIL import Image
import json


image_dict = {}
label_dict = {}
mask_dict = {}
stats_dict = {}
test_split = []
train_split = []


def train_test_split(full, positive, test_fraction):
    negative = full - positive
    test_neg_count = int(np.ceil(len(negative)*test_fraction))
    test_pos_count = int(np.ceil(len(positive)*test_fraction))
    negative_list = list(negative)
    positive_list = list(positive)
    np.random.shuffle(positive_list)
    np.random.shuffle(negative_list)
    test_positive = set()
    for i in range(test_pos_count):
        test_positive |= set([positive_list[i]])
    train_positive = positive - test_positive
    if test_neg_count > 1:
        test_negative = set()
        for i in range(test_neg_count):
            test_negative |= set([negative_list[i]])
        train_negative = negative - test_negative
        train = list(train_positive | train_negative)
        test = list(test_positive | test_negative)
    else:
        train = list(train_positive)
        test = list(test_positive)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train, test)


def load_image(root, series):
    
    img_file = series + ".png"

    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    y, x = np.shape(img)
    img = img.reshape((1, y, x))
    return img

def load_label(root, series):
    
    img_file = series + ".png"
    json_file = series + ".json"
    
    im = sitk.ReadImage(img_file)
    im_array = sitk.GetArrayFromImage(im)
    columns = len(im_array)
    rows = len(im_array[1])

    img = np.zeros((rows, columns))
    labels_n_points = obtainpoints(json_file) 
    
    for label in labels_n_points:
        points = labels_n_points[label]
        
        r = []
        c = []

        for i in range(0,len(points)):
            r.append(points[i][0])
            c.append(points[i][1])       

        rr, cc = polygon(r, c)

        img[rr, cc] == 1
        
    return img

def obtainpoints(json_file):
    label_n_points = dict()

    x = open(json_file)
    json_file2 = json.load(x)
    x.close()

    nlabels = len(json_file2["shapes"])
    for i in range(0,nlabels):
        label = json_file2["shapes"][i]["label"]
        points = json_file2["shapes"][i]["points"]

        label_n_points[label] = points
    
    return label_n_points

def full_dataset(root_dir, images):
    image_list = []
    image_files = []

    for (root,dirs,files) in os.walk(root_dir):
            root_embroyo = root
            files_embroyo = files
            
            for i in range(0,len(files_embroyo)):
                if(files_embroyo[i] == 'metadata.json'):
                    embroyo_metadata = os.path.join(root_embroyo,files_embroyo[i])
                    channel = obtainchannel(embroyo_metadata)
                    image_directory = os.path.join(root_embroyo, channel) + "/IntensityImages"                  
                    image_file_list = glob.glob(image_directory + "/*png")

                    for img in image_file_list:
                        base = os.path.splitext(img)[0]
                        image_list.append(base)
    
    return image_list

def make_dataset(root_dir, images, targets, seed, train, class_balance, partition, nonempty, test_fraction, mode):
    global image_dict, label_dict, test_split, train_split

    zero_tensor = None

    train = mode == "train"
    label_list = []

    for (root,dirs,files) in os.walk(root_dir):
        
        for i in range(0,len(files)):
            if(files[i] == 'metadata.json'):
                embroyo_metadata = os.path.join(root,files[i])
                channel = obtainchannel(embroyo_metadata)
                image_directory = os.path.join(root, channel) + "/IntensityImages"                  
                json_file_list = glob.glob(image_directory + "/*json")

                for anno in json_file_list:
                    base = os.path.splitext(anno)[0]
                    label_list.append(base)

    zero = label_list[1]
    sample_label = load_label(root_dir, zero)
    shape = np.shape(sample_label)
    
    if len(test_split) == 0:
        zero_tensor = np.zeros(shape, dtype=np.uint8)
        image_list = []
        
        file_list=[]

        for (root,dirs,files) in os.walk(root_dir):
            root_embroyo = root
            files_embroyo = files
            
            for i in range(0,len(files_embroyo)):
                if(files_embroyo[i] == 'metadata.json'):
                    embroyo_metadata = os.path.join(root_embroyo,files_embroyo[i])
                    channel = obtainchannel(embroyo_metadata)
                    image_directory = os.path.join(root_embroyo, channel) + "/IntensityImages"                  
                    image_file_list = glob.glob(image_directory + "/*png")

                    for img in image_file_list:
                        base = os.path.splitext(img)[0]
                        file_list.append(base)               

        image_list = label_list

        np.random.seed(seed)
        full = set(image_list)
        positives = set(label_list) & full
        train_split, test_split = train_test_split(full, positives, test_fraction)
    
    if train:
        print(len(label_list))
        print(len(train_split))
        keys = train_split
    else:
        print(len(test_split))
        keys = test_split
    
    y, x = shape

    result = []
    target_means = []

    for key in keys:
        target = load_label(root_dir, key)
        target_means.append(np.mean(target))
        result.append(key)

    target_mean = np.mean(target_means)
    return (result, target_mean)

def obtainchannel(metad):
    x = open(metad)
    metad = json.load(x)
    x.close()

    for key in metad:
        if key[:-1] == "Channel":
            if metad[key]["Stain"] == "PMC":
                channel = key    

    return (channel)

class PMC_Dataset(data.Dataset):
    def __init__(self, root='.', images=None, targets=None, transform=None,
                 target_transform=None, co_transform=None, mode="train", seed=1,
                 class_balance=False, split=None, masks=None, nonempty=True,
                 test_fraction=0.25):

        self.mode = mode
        self.root = root

        if masks is not None:
            self.masks = os.path.join(self.root, masks)
        if targets is not None:
            self.targets = os.path.join(self.root)
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        if images is None:
            raise(RuntimeError("images must be set"))
        if targets is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))
        if mode == "infer":
            imgs = full_dataset(root, images)
        else:
            imgs, target_mean = make_dataset(root, images, targets, seed, mode, 
                                             class_balance, split, nonempty,
                                             test_fraction, mode)
            self.data_mean = target_mean
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images: " + os.path.join(root + "\n")))

        
        self.imgs = imgs
        self.masks = None
        self.split = split

    def target_mean(self):
        return self.data_mean

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "test":
            return self.__getitem_dev(index)
        elif self.mode == "infer":
            return self.__getitem_prod(index)

    def __getitem_prod(self, index):
        series = self.imgs[index]
        image = load_image(self.images, series)
        origin, spacing = stats_dict[series]
        image = image.astype(np.float32)
        if self.split is not None:
            batches = utils.partition_image(image, self.split)
        else:
            batches = [image]
        if self.transform is not None:
            batches = map(self.transform, batches)
            batches = [batches]
        batches = torch.cat(batches)
        return batches, series, origin, spacing

    def __getitem_dev(self, index):
        series = self.imgs[index]
        target = load_label(self.root, series)
        target = torch.from_numpy(target.astype(np.int64))
        img = load_image(self.images, series)        
        

        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.co_transform is not None:
            img, target = self.co_transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

