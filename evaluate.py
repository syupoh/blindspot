import os
import glob
import json
import random
import pprint
import argparse

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utils import spot2label, label2spot, plot_confusion_matrix, plot_prediction
from models import select_model, grad_cam
from dataset import EGDSpotDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import timm

SEED = 41
WEIGHTS = None

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', type=str, help='select gpu to train', default="0")
    parser.add_argument('--val_data', type=str, help='base path of data to evaluate', 
    default="{0}/dataset/endoscopy/JFD-03K/d1.9/".format(os.getcwd()[:os.getcwd().find('/python/')]
    ))
    parser.add_argument('--save_dir', type=str, help='directory to save grad cam images', 
    default="{0}/dataset/endoscopy/JFD-03K/incorrects/".format(os.getcwd()[:os.getcwd().find('/python/')]
    ))
    # parser.add_argument('--normalize', type=str2bool, help='normalize in preprocess', default="False")
    parser.add_argument('--scale', action="store_true", help='scale in preprocess')
    parser.add_argument('--label_smoothing', "-ls", type=float, help='value of label smoothing', default=0.1)
    parser.add_argument('--ckpt_path', type=str, help='checkpoint path to load', 
    default="{0}/dataset/endoscopy/ckpt/xception_extractor/cv_d1.4_scale/best_cnn_20210314.h5".format(os.getcwd()[:os.getcwd().find('/python/')]
    ))
    parser.add_argument('--batch', type=int, default=4, help='set batch size')
    parser.add_argument("--cam", action="store_true", help="generate Grad CAM result images")
    parser.add_argument("--plot", action="store_true", help="plot prediction results")
    
    # parser.add_argument("--collect_wrong", action="store_true", help="collect list of wrong prediction images")

    return parser.parse_args()


def rival(idx, cm):
	tp, tn, fp, fn = 0, 0, 0, 0
    # specificity
	for i in range(len(cm)):
		if i != idx:
			fp += cm[i][idx]
			tn += cm[i][i]

    # sensitivity
	for i in range(len(cm[idx])):
		if i == idx:
			tp += cm[idx][i]
		else:
			fn += cm[idx][i]
	return tp, tn, fp, fn


def get_metrics(cm, labels):
	results = dict()
	for i in range(len(cm)):
		label = labels[i]
		tp, tn, fp, fn = rival(i, cm)
		sensitivity = tp / (tp + fn)
		specificity = tn / (tn + fp)
		results[label] = {
			"Sensitivity": round(sensitivity, 4),
			"Specificity": round(specificity, 4),
		}
	return results


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def evaluate(model, dataset, save_dir, cam, plot= False):
    @tf.function
    def val_step(batch_x, batch_y):
        predictions = model(batch_x)
        loss = loss_function(batch_y, predictions)
        val_loss(loss)
        val_acc(batch_y, predictions)
        return predictions

    loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    val_loss = tf.keras.metrics.Mean()
    val_acc = tf.keras.metrics.CategoricalAccuracy()

    confidence = list()

    desc = "Evaluation Step"
    t = tqdm(enumerate(dataset), desc=desc)
    for i, (batch_x, batch_y) in t:
        predictions = val_step(batch_x, batch_y)
        confidence += predictions.numpy().tolist()

        t.set_description(desc + f" {(i+1)}/{len(dataset)}")
        loss = val_loss.result().numpy()
        acc = val_acc.result().numpy()
        t.set_postfix({
            "Evaluation Loss": round(loss, 4),
            "Evaluation Accuracy": round(acc, 4)
        })

    confidence = np.array(confidence)
    acc = float(val_acc.result().numpy())
    loss = float(val_loss.result().numpy())
    pred_class = np.argmax(confidence, axis=1)
    
    y_true = np.array(dataset.y).argmax(axis=1)
    files = np.array(dataset.x)
    cm = confusion_matrix(y_true, pred_class)
    labels = list(spot2label.keys())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(cm, target_names=labels, path=os.path.join(save_dir, "Confusion Matrix"))
    metrics = get_metrics(cm, labels)
    metrics["Total Accuracy"] = round(acc, 4)
    metrics["Loss"] = round(loss, 4)
    dump_json(os.path.join(save_dir, "results.json"), 
        metrics)

    if cam:
        idxs = np.arange(len(pred_class))[pred_class != y_true]
        dataset.batch_size = 1
        paths = dataset.x[idxs] 
        images = [dataset[i][0] for i in idxs]
        preds = pred_class[idxs]
        gts = y_true[idxs]
        confidences = confidence[idxs]
        grad_cam(list(zip(paths, images, preds, gts, confidences)), save_dir, model, scale=dataset.scale)

    if plot:
        plot_prediction(list(zip(dataset.x, y_true, confidence)))

def get_data(root_path):
    X, y = [], []
    for path, dirs, files in os.walk(root_path):
        if not dirs:
            for file in files:
                dir_name = os.path.basename(path)
                X.append(os.path.join(path, file))
                y.append(spot2label[dir_name])
    return np.array(X), np.array(y)

def classify(model, dataset):
    files = np.array(dataset.x)
    predicts = model.predict(dataset, verbose=1)
    predicts_sort = predicts.argsort(axis=1)

    confidences, spots = [], []
    for i in range(len(predicts)):
        nlarge_idx = np.flip(predicts_sort[i][-3::])
        nlarge_preds = predicts[i][nlarge_idx]
        nlarge_spots = np.array(list(map(lambda x: label2spot[x], nlarge_idx)))
        confidences.append(tuple(nlarge_preds))
        spots.append(tuple(nlarge_spots))

    return zip(files, confidences, spots)


if __name__ == "__main__":
    args = argparser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    image_shape = (299, 299, 3)
    model_name = "xception"

    dataset = EGDSpotDataset.initialize_from_path(args.val_data, batch_size=args.batch, image_shape=image_shape, scale=args.scale)

    # model = select_model(model=model_name, weights=WEIGHTS)
    # model.load_weights(args.ckpt_path)
    model = timm.create_model('xception',num_classes =26, 
    checkpoint_path='{0}/dataset/endoscopy/JFD-03K/ckpt/20210817-131208-xception-299/model_best.pth.tar'.format(os.getcwd()[:os.getcwd().find('/python/')]
    ))
    results = evaluate(model, dataset, args.save_dir, cam=args.cam)
