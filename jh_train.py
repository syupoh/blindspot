import os, sys
import glob
import json
import random
import pprint
import argparse

import pprint
import config

import numpy as np
import tensorflow as tf

from jh_models import Xception
from evaluate import evaluate, dump_json

from tqdm import tqdm, trange
from datetime import datetime
from utils import draw_graph, plot_confusion_matrix, draw_fold_graph, spot2label, label2spot, plot_confusion_matrix, plot_prediction
from models import adaptive_clip_grad, grad_cam
from dataset import EGDSpotDataset, spot2label, get_data

from tfdeterminism import patch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from syslogger.syslogger import Logger

import pdb

SEED = 41
PATIENCE = 10
WEIGHTS = None

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', type=str, help='select gpu to train', default=3)
    parser.add_argument('--train_path', "-t", type=str, help='base path to train set')
    parser.add_argument('--val_path', "-v", type=str, help='base path to validation set', default = "")
    #parser.add_argument('--scale', action="store_true", help='scale in preprocess')
    parser.add_argument('--save_path', "-s", type=str, help='base path to save', default = "")
    parser.add_argument('--label_smoothing', "-ls", type=float, help='value of label smoothing', default=0.1)
    #parser.add_argument('--ckpt_name', type=str, default="", help='checkpoint name to save model')
    # parser.add_argument('--augmentation', action="store_true", help='use augmentation data')
    parser.add_argument('--cross_validation', "-cv", action="store_true", help='Do Cross Validation')
    parser.add_argument('--batch', type=int, default=config.BATCH, help='set batch size')
    #parser.add_argument('--evaluate', action="store_true", help='evaluate train results')
    return parser.parse_args()


def save_dataset(dataset, path):
    with open(path, "w") as f:
        for data in list(set(dataset.x)):
            f.write(data+"\n")


def fit(model, args, train_dataset, val_dataset, save_path, best_acc=-1):
    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = loss_function(batch_y, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_loss(loss)
        train_acc(batch_y, predictions)

    @tf.function
    def val_step(batch_x, batch_y):
        predictions = model(batch_x)
        loss = loss_function(batch_y, predictions)
        val_loss(loss)
        val_acc(batch_y, predictions)
        top_k_acc(batch_y, predictions)

    def reset():
        train_acc.reset_states()
        train_loss.reset_states()
        val_acc.reset_states()
        val_loss.reset_states()
        top_k_acc.reset_states()

    n_fold, n_epoch, lr, label_smoothing = args
    save_dir = os.path.dirname(save_path)

    desc = "Train Step"
    max_acc, final_loss = -1, 0
    patient_cnt = 0
    factor = 1e-1
    min_lr = 1e-6

    loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    # base_optim = tf.keras.optimizers.Nadam(lr=lr)
    # optimizer = SAM(base_optim)
    optimizer = tf.keras.optimizers.Nadam(lr=lr)
    # optimizer = tf.keras.optimizers.Adam(lr=lr)


    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()

    train_acc = tf.keras.metrics.CategoricalAccuracy()
    val_acc = tf.keras.metrics.CategoricalAccuracy()
    top_k_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

    history = {
        "accuracy": list(),
        "val_accuracy": list(),
        "top5_acc": list(),
        "loss": list(),
        "val_loss": list(),
    }

    t = trange(n_epoch, desc=desc, leave=True)
    for epoch in t:

        # train
        for i, (batch_x, batch_y) in enumerate(train_dataset):
            t.set_description(f"- Fold {n_fold} Epoch {epoch+1}/{n_epoch} | " + desc + f" {(i+1)}/{len(train_dataset)}")
            train_step(batch_x, batch_y)

            
            loss = train_loss.result().numpy()
            acc = train_acc.result().numpy()
            t.set_postfix({
                "Train Loss": round(loss, 4),
                "Train Accuracy": round(acc, 4)
            })
            t.refresh()
        history["loss"].append(loss)
        history["accuracy"].append(acc)

        # validation
        for batch_x, batch_y in val_dataset:
            val_step(batch_x, batch_y)
        
        val_acc_res = val_acc.result().numpy()
        val_loss_res = val_loss.result().numpy()
        top_k_acc_res = top_k_acc.result().numpy()
        history["val_accuracy"].append(val_acc_res)
        history["val_loss"].append(val_loss_res)
        history["top5_acc"].append(top_k_acc_res)

        print(f"- test loss: {val_loss_res:.4f} - test accuracy: {val_acc_res:.4f} - test top-5 accuracy: {top_k_acc_res:.4f}")

        if max_acc < val_acc_res:
            patient_cnt = 0
            max_acc = val_acc_res
            final_loss = val_loss_res

            fold_cnn_path = os.path.join(save_dir, f"fold_{n_fold}_cnn_"+os.path.basename(save_path))
            print(f"Attain Best Accuracy: {max_acc:.4f}")
            print(f"Save Fold {n_fold} Model to", fold_cnn_path)
            model.save_weights(fold_cnn_path)
            if best_acc < max_acc:
                best_acc = max_acc
                model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
                best_cnn_path = os.path.join(save_dir, "best_cnn_"+os.path.basename(save_path))
                model.save_weights(best_cnn_path)
                print(f"Save Best Accuracy CNN Model {best_acc:.4f} to", best_cnn_path)

                evaluate(model, val_dataset, save_dir, cam=False)

        else:
            patient_cnt += 1
            if PATIENCE == patient_cnt:
                print("Stop training since the model doesn't improve")
                break
            elif patient_cnt % 4 == 0:
                lr = max(lr * factor, min_lr)
                optimizer.lr.assign(lr)
                # base_optim.lr.assign(lr)
                # optimizer.base_optimizer = base_optim

        reset()
    draw_graph(history, os.path.join(save_dir, f"result per epoch"))
    return max_acc, final_loss


def main():
    args = argparser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # save_path = f"/data/syupoh/dataset/endoscopy/JFD-03K/ckpt/Xception/{datetime.now().strftime('%Y%m%d-%H%M%S')}.h5"
    # ckpt_name = os.path.basename(save_path)
    # save_dir = os.path.dirname(save_path)
    
    dataname = args.train_path.split('/jfd_dataset/')[-1].split('/')[0] # timm3
    curtime = '{0}'.format(datetime.now().strftime('%Y%m%d_%H%M'))[2:] # 211102_1600

    save_dir = '{0}/{1}'.format(args.save_path, dataname) # ckpt/Xception/timm3
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_dir = '{0}/{1}'.format(save_dir, curtime) # ckpt/Xception/timm3/211102_1600
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = '{0}/{1}_{2}.h5'.format(save_dir, dataname, curtime) # ckpt/Xception/timm3/211102_1600/timm3_211102_1600.h5
    sys.stdout = Logger('{0}/{1}_{2}.txt'.format(save_dir, dataname, curtime)) # ckpt/Xception/timm3/211102_1600/timm3_211102_1600.txt
    
    print(args)

    image_shape = (299, 299, 3)
    model_name = "xception"
    
    X, y = get_data(args.train_path)
    fold = 5
    train, test = [], []
    if args.cross_validation:
        kfold = KFold(n_splits=fold, shuffle=True, random_state=SEED)
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_dataset = EGDSpotDataset.initialize_from_data(X_train.tolist(), y_train.tolist(), args.batch, image_shape)    
            test_dataset = EGDSpotDataset.initialize_from_data(X_test, y_test, args.batch, image_shape,  scale=True, train=False)    
            train.append(train_dataset)
            test.append(test_dataset)
    else:
        if args.val_path == "":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y, shuffle=True) 
            train_dataset = EGDSpotDataset.initialize_from_data(X_train.tolist(), y_train.tolist(), args.batch, image_shape)    
            test_dataset = EGDSpotDataset.initialize_from_data(X_test, y_test, args.batch, image_shape, scale=True, train=False)    
            train.append(train_dataset)
            test.append(test_dataset)
        else:
            X_test, y_test = get_data(args.val_path)
            train_dataset = EGDSpotDataset.initialize_from_data(X.tolist(), y.tolist(), args.batch, image_shape)    
            test_dataset = EGDSpotDataset.initialize_from_data(X_test.tolist(), y_test.tolist(), args.batch, image_shape, scale=True, train=False)    
            train.append(train_dataset)
            test.append(test_dataset)


    # K-Fold Cross Validation
    args = [-1, config.EPOCH, config.LR, args.label_smoothing]

    best_acc = -1
    losss, accs = [], []
    for n_fold, (train_dataset, test_dataset) in enumerate(zip(train, test)):
        args[0] = n_fold + 1
        model = Xception()
        max_acc, val_loss = fit(model, args, train_dataset, test_dataset, save_path, best_acc)
        best_acc = max(best_acc, max_acc)
        accs.append(max_acc)
        losss.append(val_loss)
        del model
    print(f"{len(accs)} Fold CV | - average loss: {np.average(losss):.4f} - average accuracy: {np.average(accs):.4f} - best accuracy: {best_acc:.4f}")
    draw_fold_graph(accs, os.path.join(save_dir, "result"))

if __name__ == "__main__":
    main()
    

