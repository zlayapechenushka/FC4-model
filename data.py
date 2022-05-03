import os
import cv2
import scipy.io
import torch
import numpy as np
import torch.utils.data as data

from utils import hwc_to_chw, linear_to_nonlinear, bgr_to_rgb, normalize
class ColorCheckerDataset(data.Dataset):

    def __init__(self, train = True, folds_num = 1):

        self.__train = train
  

        path_to_folds = os.path.join("dataset", "folds.mat")
        path_to_metadata = os.path.join("dataset", "metadata.txt")
        self.__path_to_data = os.path.join("dataset", "preprocessed", "numpy_data")
        self.__path_to_label = os.path.join("dataset", "preprocessed", "numpy_labels")

        folds = scipy.io.loadmat(path_to_folds)
        img_idx = folds["tr_split" if self.__train else "te_split"][0][folds_num][0]

        metadata = open(path_to_metadata, 'r').readlines()
        self.__fold_data = [metadata[i - 1] for i in img_idx]

    def __getitem__(self, index):
        file_name = self.__fold_data[index].strip().split(' ')[1]
        img = np.array(np.load(os.path.join(self.__path_to_data, file_name + '.npy')), dtype='float32')
        illuminant = np.array(np.load(os.path.join(self.__path_to_label, file_name + '.npy')), dtype='float32')

        if self.__train:
            img, illuminant = img, illuminant
        else:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

        img = hwc_to_chw(linear_to_nonlinear(bgr_to_rgb(normalize(img))))

        img = torch.from_numpy(img.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        if not self.__train:
            img = img.type(torch.FloatTensor)

        return img, illuminant, file_name
   
    def __len__(self):
        return len(self.__fold_data)