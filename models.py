import os
import pickle
import config
from config import IMAGE_SHAPE

import imutils
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, applications, Sequential, Model
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from utils import label2spot, spot2label

###############################################################################################################################
# Base Models
###############################################################################################################################
def EfficientNetB7(weights, input_shape):
    base_model = applications.EfficientNetB7(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


def EfficientNetB0(weights, input_shape):
    base_model = applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


def VGG19(weights, input_shape):
    base_model = applications.VGG19(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


# def XceptionNet(weights, input_shape):
#     base_model = applications.Xception(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
#     base_model.trainable = False if weights == "imagenet" else True
#     output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
#     model = Model(base_model.input, output)
#     return model

def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    # kernel_std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)



def XceptionNet(weights, input_shape):
    base_model = applications.Xception(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)

    # Weight Standardization (comment last two lines to apply only gn) + Group Normalization
    for i, layer in enumerate(model.layers):
        if layer.name[-2:] == "bn":
            model.layers[i] = tfa.layers.GroupNormalization()
            model.layers[i]._name = layer.name[:-2] + "gn"
        elif layer.name[-5:-1] == "conv":
            layer.kernel_regularizer = ws_reg
            # layer.kernel_initializer = tf.keras.initializers.HeNormal()

    return model


def XceptionNetExtractor(weights, input_shape):
    base_model = applications.Xception(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    return Model(inputs=[base_model.input], outputs=[base_model.layers[-1].output, output])


def DenseNet(weights, input_shape):
    base_model = applications.DenseNet169(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True    
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


def InceptionResNet(weights, input_shape):
    base_model = applications.InceptionResNetV2(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True    
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


def ResNet101V2(weights, input_shape):
    base_model = applications.ResNet101V2(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True    
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


def ResNet50V2(weights, input_shape):
    base_model = applications.ResNet50V2(input_shape=input_shape, include_top=False, weights=weights, pooling="avg")
    base_model.trainable = False if weights == "imagenet" else True    
    output = layers.Dense(config.NUM_CLASS, activation='softmax', name="softmax")(base_model.output)
    model = Model(base_model.input, output)
    return model


def select_model(**kwargs):
    if kwargs["model"] == "vgg19":
        return VGG19(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "resnet50v2":
        return ResNet50V2(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "resnet101v2":
        return ResNet101V2(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "efficientnetb0":
        return EfficientNetB0(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "efficientnetb7":
        return EfficientNetB7(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "densenet169":
        return DenseNet(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "inception_resnet":
        return InceptionResNet(weights=kwargs["weights"], input_shape=config.IMAGE_SHAPE)
    elif kwargs["model"] == "xception":
        return XceptionNet(weights=kwargs["weights"], input_shape=(299, 299, 3))
    elif kwargs["model"] == "xception_extractor":
        return XceptionNetExtractor(weights=kwargs["weights"], input_shape=(299, 299, 3))
    else:
        raise NotImplementedError


def compile_model(model_name, weights, lr=config.LR, label_smoothing=0.1):
    model = select_model(model=model_name, weights=weights)
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing),
        metrics=['accuracy'])
    return model


###############################################################################################################################
# Auxiliary models
###############################################################################################################################
def grad_cam(zips, save_dir, model, scale):
    """#5.1 Grad-CAM 이미지 생성 함수

    Args:
        fpaths (list): Grad-CAM 이미지를 생성할 파일 경로들
        save_dir (String): 저장할 디렉토리 경로
        models (tuple): (tf.keras.Model, 모델 이름)
    """
    for path, image, pred, gt, confidence in tqdm(zips, total=len(zips), desc="Generate CAM Image"):
        origin = cv2.imread(path)
        outputs = []
        for idx in [pred, gt]:
            cam = GradCAM(model, class_idx=idx)
            heatmap = cam.compute_heatmap(image)

            heatmap = cv2.resize(heatmap, (origin.shape[1], origin.shape[0]))
            _, output = cam.overlay_heatmap(heatmap, origin, alpha=0.5)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            outputs.append(output)

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        plt.subplot(131)
        plt.title(f"original image: {label2spot[gt]}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(origin)

        plt.subplot(132)
        plt.title(f"prediction: {label2spot[pred]} {confidence[pred] * 100:.2f}%")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(outputs[0])

        plt.subplot(133)
        plt.title(f"ground truth: {label2spot[gt]} {confidence[gt] * 100:.2f}%")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(outputs[1])
        
        spot = os.path.basename(label2spot[gt])
        dir_path = os.path.join(save_dir, spot)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        plt.savefig(os.path.join(dir_path, os.path.splitext(os.path.basename(path))[0]))
        plt.close()

class GradCAM:
    """ #4.1 CNN 네트워크가 추론한 이미지에 대한 Grad-CAM 이미지 생성
    """
    def __init__(self, model, class_idx, layer_name=None):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()


    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output,
                self.model.output[-1]])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[self.class_idx]
        grads = tape.gradient(loss, conv_outputs)

        castconv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = castconv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap


    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_TURBO):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)


""" An implementation of Adaptive Gradient Clipping
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
Code references:
  * Official JAX implementation (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
  * Ross Wightman's implementation https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
"""
def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in {2, 3}:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01,
                       eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads

"""
SAM
"""

class SAM():
    def __init__(self, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.rho = rho
        self.base_optimizer = base_optimizer

    def first_step(self, gradients, trainable_variables):
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(trainable_variables)
        for i in range(len(trainable_variables)):
            e_w = gradients[i] * self.rho / (grad_norm + 1e-12)
            trainable_variables[i].assign_add(e_w)
            self.e_ws.append(e_w)


    def second_step(self, gradients, trainable_variables):
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])
        # do the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))

    # if you want to use model.fit(), override the train_step method of a model with this function, example is mnist_example_keras_fit.
    # for customization see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
    def sam_train_step(self, data, rho=0.05):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # first step
        e_ws = []
        grad_norm = tf.linalg.global_norm(trainable_vars)
        for i in range(len(trainable_vars)):
            e_w = gradients[i] * rho / (grad_norm + 1e-12)
            trainable_vars[i].assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        for i in range(len(trainable_vars)):
            trainable_vars[i].assign_add(-e_ws[i])
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}