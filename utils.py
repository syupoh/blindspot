import os
import cv2
import pickle
import itertools

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from config import EPOCH

spot2label = {
    'GEJ': 0,
    'LB-AW': 1,
    'LB-GC': 2,
    'LB-LC': 3,
    'LB-PW': 4,
    'angle-AW': 5,
    'angle-LC': 6,
    'angle-PW': 7,
    'antrum-AW': 8,
    'antrum-GC': 9,
    'antrum-LC': 10,
    'antrum-PW': 11,
    'cardia-AW': 12,
    'cardia-GC': 13,
    'cardia-LC': 14,
    'cardia-PW': 15,
    'duodenum bulb': 16,
    'duodenum second': 17,
    'esophagus': 18,
    'midbody-AW': 19,
    'midbody-AW (U)': 20,
    'midbody-GC': 21,
    'midbody-LC': 22,
    'midbody-LC (U)': 23,
    'midbody-PW': 24,
    'midbody-PW (U)': 25
}

label2spot = {i:k for k, i in spot2label.items()}

# def save_result(path):
#     with open(path, "w") as f:
#         for k, v in results.items():
#             f.write(f"- {k} loss: {v[0]:.4f}, acc: {v[1]:.4f}\n")


def get_file_paths(path):
    ret = []
    for path, dirs, files in os.walk(path):
        if not dirs:
            for file in files:
                ret.append(os.path.join(path, file))
    return ret


def draw_fold_graph(dl_accs, save_path):
    index = np.arange(len(dl_accs))
    xticks = [f"{i+1}" for i in range(len(index))]
    ylim = [0.5, 1]
    fontsize = 18
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    
    width = 0.3
    bars = plt.bar(index, dl_accs, label='CNN', width=width, alpha=0.9)
    plt.axhline(y=np.average(dl_accs), label="CNN Average", color="r", linewidth=2)
    
    for i, b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()*0.5, b.get_height()+0.001, round(dl_accs[i], 4), ha="center", fontsize=12)
        
        
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.xticks(index, xticks, fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.xlabel('Fold', fontsize=fontsize)
    plt.ylim(ylim)
    plt.yticks(fontsize=fontsize)
    plt.title('Validation Accuracy', fontsize=fontsize)
    # plt.show()
    fig.savefig(save_path)
    plt.close()


def draw_graph(history, save_path):
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    index = np.arange(len(acc))
    xticks = [f"{i+1}" for i in range(len(acc))]
    ylim = [0.2, 1]

    # plt.figure(figsize=(12, 6))
    plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xticks(index, xticks, fontsize=8)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    # plt.ylim([min(plt.ylim()),1])
    plt.ylim(ylim)
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy Loss')
    plt.ylim([0,5.0])
    plt.xticks(index, xticks, fontsize=8)
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(save_path)
    plt.close()

    
def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, path='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    
# qt.qpa.screen: QXcbConnection: Could not connect to display localhost:11.0
# Could not connect to any X display.

    # print('plot????')
    plt.figure()
    # plt.figure(figsize=(12, 8))
    # print('plot?')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(os.path.basename(path))
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path)
    plt.close()


def imagenet_transform(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    return (img - mean) / std


def scale(img, scaling):
    if scaling:
        return img / 255.0
    else:
        return img

def plot_prediction(zip):
    for path, gt, confidence in tqdm(zip):
        image = cv2.imread(path)
        fig, axes = plt.subplots(1,2,figsize=(20,10))

        plt.subplot(121)
        plt.title(f"image: {label2spot[gt]}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)

        # TODO
        plt.subplot(122)
        plt.title(f"prediciton results")
        plt.xticks(range(10))
        plt.yticks([])
        plt.imshow()

