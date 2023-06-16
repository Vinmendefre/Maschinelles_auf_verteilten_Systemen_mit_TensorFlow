import math
from typing import List

import tensorflow as tf
import argparse
import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split


# Erstellen des Modells
class ArcMarginProduct(tf.keras.layers.Layer):
    """
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    """

    def __init__(
            self, n_classes, s=30, m=0.50, easy_margin=False, ls_eps=0.0, **kwargs
    ):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m": self.m,
                "ls_eps": self.ls_eps,
                "easy_margin": self.easy_margin,
            }
        )
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer="glorot_uniform",
            dtype="float32",
            trainable=True,
            regularizer=None,
        )

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1), tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(tf.one_hot(y, depth=self.n_classes), dtype=cosine.dtype)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def get_arc_model(
        n_classes: int,
        image_shape=(300, 300),
        learning_rate: float = 0.001,
        dense_layers: List[int] = [],
):
    """Create Convnet Model with ArcMargin Layer."""
    margin = ArcMarginProduct(
        n_classes=n_classes, s=30, m=0.5, name="head/arc_margin", dtype="float32"
    )

    inp = tf.keras.layers.Input(shape=(*image_shape, 3), name="inp1")
    label = tf.keras.layers.Input(shape=(), name="inp2")

    x = tf.keras.applications.EfficientNetB3(weights="imagenet", include_top=False)(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for dense in dense_layers:
        x = tf.keras.layers.Dense(dense)(x)

    x = margin([x, label])

    output = tf.keras.layers.Softmax(dtype="float32")(x)

    model = tf.keras.models.Model(inputs=[inp, label], outputs=[output])

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


# Laden und Vorbereiten der Daten
def _load_data(dat_dir):
    if os.path.exists(os.path.join(dat_dir, 'train_images.npy')) or os.path.exists(
            os.path.join(dat_dir, 'train_labels.npy')):
        # load the data from numpy file if it exists
        print("Loading from, npy files")
        images = np.load(os.path.join(dat_dir, 'train_images.npy'))
        labels = np.load(os.path.join(dat_dir, 'train_labels.npy'))

        # return images, labels
        return train_test_split(images, labels, test_size=0.2, random_state=42)
        # return train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    print("No existing npy detected, loading data from directory")
    # if a npy file is missing they images and csv will get read out and saved
    csv_path = os.path.join(dat_dir, 'test.csv')  # TODO change this to train once testing is done
    data_df = pd.read_csv(csv_path)

    images = []
    labels = []

    image_dir = os.path.join(dat_dir, 'test_images')

    for index, row in data_df.iterrows():
        image_path = os.path.join(image_dir, row['image'])
        image = load_img(image_path, target_size=(300, 300))
        image_array = img_to_array(image)
        images.append(image_array)
        labels.append(row['label_group'])

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="float32")
    print(images[0])

    np.save(os.path.join(dat_dir, 'train_images.npy'), images, allow_pickle=True)
    np.save(os.path.join(dat_dir, 'train_labels.npy'), labels, allow_pickle=True)

    # der Random State bleibt gleich um möglichst gleiche Testbedingungen zu schaffen
    return train_test_split(images, labels, test_size=0.2, random_state=42)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=1)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    root_path = '/root/datasets'
    _load_data(root_path)

    train_images, test_images, train_labels, test_labels = _load_data(args.train)
    n_classes = max(max(train_labels), max(test_labels)) + 1

    print('Training model for {} epochs..\n\n'.format(args.epochs))

    model = get_arc_model(n_classes=n_classes, dense_layers=[5, 5])
    model.fit(train_images, train_labels, epochs=args.epochs)
    model.evaluate(test_images, test_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
