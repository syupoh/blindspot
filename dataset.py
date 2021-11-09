import os
import math
import random
import config

import numpy as np
import tensorflow as tf

from utils import scale, spot2label
from matplotlib.image import imread
from tensorflow.keras.utils import Sequence


SEED = 41
PATIENCE = 4
WEIGHTS = None

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def shuffle(X, y):
    shuffle_idx = np.arange(len(X))
    np.random.shuffle(shuffle_idx)
    return  np.array(X)[shuffle_idx], np.array(y)[shuffle_idx]


def load_image(fpath, image_shape, scale_flag):
    img = imread(fpath)
    resized = tf.image.resize(img, image_shape, method=tf.image.ResizeMethod.AREA)
    return scale(resized, scale_flag)

def get_data(root_path):
    X, y = [], []
    lower_spot2label = {}
    for k, v in spot2label.items():
        lower_spot2label[k.lower()] = v

    if os.path.splitext(root_path)[-1] != ".txt":
        for path, dirs, files in os.walk(root_path):
            if not dirs:
                for file in files:
                    X.append(os.path.join(path, file))
                    # dir_name = os.path.basename(path)
                    # y.append(spot2label[dir_name])
                    dir_name = os.path.basename(path).lower()
                    y.append(lower_spot2label[dir_name])
    else:
        with open(root_path, "r") as f:
            for line in f.readlines():
                dir_name = os.path.basename(os.path.dirname(root_path))
                X.append(line)
                y.append(spot2label[dir_name])
    return np.array(X), np.array(y)

class EGDSpotDataset(Sequence):
    def __init__(self, X, y, batch_size, image_shape, scale=True):
        self.batch_size = batch_size
        self.x, self.y = X, y
        self.image_shape = image_shape
        # self.normalize = normalize
        self.scale = scale
    
    @classmethod
    def initialize_from_data(cls, X, y, batch_size, image_shape, scale=True, train=True):
        if train:
            X, y = shuffle(X, y)
        onehot_y = []
        for label in y:
            onehot = [0] * config.NUM_CLASS
            onehot[label] = 1
            onehot_y.append(onehot)
        return cls(X=X, y=onehot_y, batch_size=batch_size, image_shape=image_shape, scale=scale)
    
    
    @classmethod
    def initialize_from_path(cls, root_path, batch_size, image_shape, scale=True):
        X, y = [], []
        if os.path.splitext(root_path)[-1] == ".txt":
            with open(root_path, "r") as f:
                for fpath in f.readlines():
                    fpath = fpath.strip()
                    dir_name = os.path.basename(os.path.dirname(fpath.strip()))
                    X.append(fpath)
                    onehot = [0] * len(spot2label)
                    onehot[spot2label[dir_name]] = 1
                    y.append(onehot)

        else:
            for path, dirs, files in os.walk(root_path):
                if not dirs:
                    for file in files:
                        dir_name = os.path.basename(path)
                        X.append(os.path.join(path, file))
                        onehot = [0] * len(spot2label)
                        onehot[spot2label[dir_name]] = 1
                        y.append(onehot)

        return cls(X=np.array(X), y=np.array(y), batch_size=batch_size, image_shape=image_shape, scale=scale)
    

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        return tf.convert_to_tensor([load_image(fpath, (self.image_shape[0], self.image_shape[1]), self.scale) for fpath in batch_x]), tf.convert_to_tensor(batch_y) 


class EGDSpotTestDataset(Sequence):
    def __init__(self, root_path, batch_size, image_shape, scale):
        self.batch_size = batch_size
        self.x = self._get_file_paths(root_path)
        self.image_shape = image_shape
        self.scale = scale


    def _get_file_paths(self, root):
        ret = []
        for path, dirs, files in os.walk(root):
            if not dirs:
                for file in files:
                    ret.append(os.path.join(path, file))
        return ret

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        return tf.convert_to_tensor([load_image(fpath, (self.image_shape[0], self.image_shape[1]), self.scale) for fpath in batch_x])
