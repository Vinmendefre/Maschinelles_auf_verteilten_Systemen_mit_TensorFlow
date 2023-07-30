import abc
from abc import ABC
from datetime import time
import tensorflow as tf
import argparse
import os
import json
import pandas as pd

import math
from typing import List


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
    print(model.summary())
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

class BaseDataLoader(ABC):
    """
    Base class loading datasets

    """

    def __init__(
        self,
        base_path: str,
        filenames: List[str],
        labels: List[str],
        target_height: int,
        target_width: int,
        batch_size: int,
        shuffle_buffer_size: int,
        data_augmentation: bool,
    ):
        self.base_path = base_path
        self.filenames = filenames
        self.labels = labels
        self.target_height = target_height
        self.target_width = target_width
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.data_augmentation = data_augmentation

    @abc.abstractmethod
    def get_dataset(self) -> tf.data.Dataset:
        """
        Abstract method for training dataset creation
        """
        pass

    def get_evaluation_dataset(self) -> tf.data.Dataset:
        """Create data input pipeline."""

        load_fn = self._get_load_img_fn(
            base_path=self.base_path,
            target_height=self.target_height,
            target_width=self.target_width,
            data_augmentation=False,
        )

        def decode_image(filename):
            """Load and preprocess image"""
            num_retries = 10
            for _ in range(num_retries):
                try:
                    return load_fn(filename)
                except Exception:
                    time.sleep(1)

        dataset = tf.data.Dataset.from_tensor_slices((self.filenames,))
        dataset = dataset.map(
            decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def _get_load_img_fn(
        self,
        base_path="/opt/ml/input/data/train", #this is just the default and usually not used
        target_height=300,
        target_width=300,
        channels=3,
        data_augmentation=True,
    ):
        """Return load image function."""

        image_processing = tf.keras.Sequential()
        image_processing.add(tf.keras.layers.Resizing(target_height, target_width))

        if data_augmentation:
            image_processing.add(
                tf.keras.layers.RandomRotation(
                    factor=0.1,
                    fill_mode="nearest",
                    interpolation="bilinear",
                )
            )
            image_processing.add(tf.keras.layers.RandomFlip("horizontal"))

        def load_img(filename):
            """Return image."""
            img = tf.io.read_file(base_path + "/" + filename)
            img = tf.io.decode_image(img, channels=channels, expand_animations=False)
            img = image_processing(img)
            return tf.cast(tf.expand_dims(img, 0), tf.uint8)[0]

        return load_img


class ArcFaceDataLoader(BaseDataLoader):
    """
    Data loader class for ArcFace architecture.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_dataset(self) -> tf.data.Dataset:
        """Create data input pipeline."""

        load_fn = self._get_load_img_fn(
            base_path=self.base_path,
            target_height=self.target_height,
            target_width=self.target_height,
            data_augmentation=self.data_augmentation,
        )

        def filename_label_to_item(filename, label):
            return (
                {
                    "inp1": load_fn(filename),
                    "inp2": label,
                },
                label,
            )

        dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        dataset = dataset.map(
            filename_label_to_item, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def get_lr_callback(batch_size):
    lr_start = 0.000001
    #lr_max = 0.000005 * batch_size #learning rate für globale batch size, Testgruppen: 2, 3, 4
    #lr_max = min((0.000005 * batch_size), 0.00128) #das begrenz die  lernrate auf maximal 0.00128, Testgruppe: 8
    lr_max = (0.000005 * batch_size) / len(args.hosts)  # das passt die learning der an die anzahl der instanzen an Testgruppe: 9
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback

def _parse_args():
    parser = argparse.ArgumentParser()

    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_known_args()



if __name__ == "__main__":
    args, unknown = _parse_args()
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    LOG_DIR = "/opt/ml/output/tensorboard"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, histogram_freq=1)

    image_column = 'id'
    label_column = 'articleType'

    data_path = '/opt/ml/input/data/train'
    batch_size = args.batch_size

    train_path = os.path.join(data_path, 'train')
    train_csv_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    train_image_paths = train_df[image_column].tolist()
    train_image_labels = train_df[label_column].tolist()

    arc_face_data_loader = ArcFaceDataLoader(
        base_path=train_path,
        labels=train_image_labels,
        filenames=train_image_paths,
        target_width=300,
        target_height=300,
        batch_size=batch_size,
        shuffle_buffer_size=50,
        data_augmentation=True)

    test_path = os.path.join(data_path, 'test')
    test_csv_path = os.path.join(data_path, 'test.csv')
    test_df = pd.read_csv(test_csv_path)
    test_image_paths = test_df[image_column].tolist()
    test_image_labels = test_df[label_column].tolist()

    test_data_loader = ArcFaceDataLoader(
        base_path=test_path,
        labels=test_image_labels,
        filenames=test_image_paths,
        target_width=300,
        target_height=300,
        batch_size=batch_size,
        shuffle_buffer_size=50,
        data_augmentation=True)

    val_path = os.path.join(data_path, 'val')
    val_csv_path = os.path.join(data_path, 'val.csv')
    val_df = pd.read_csv(val_csv_path)
    val_image_paths = val_df[image_column].tolist()
    val_image_labels = val_df[label_column].tolist()

    val_data_loader = ArcFaceDataLoader(
        base_path=val_path,
        labels=val_image_labels,
        filenames=val_image_paths,
        target_width=300,
        target_height=300,
        batch_size=batch_size,
        shuffle_buffer_size=50,
        data_augmentation=True)

    train_dataset = arc_face_data_loader.get_dataset()
    test_dataset = test_data_loader.get_dataset()
    val_dataset = val_data_loader.get_dataset()

    lr_callback = get_lr_callback(batch_size)
    n_classes = int(max(max(train_image_labels), max(test_image_labels), max(val_image_labels)) + 1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'Arc_model_11000.h5',
                                                    monitor = 'val_sparse_categorical_accuracy',
                                                    verbose = 2,
                                                    save_best_only = True,
                                                    save_weights_only = True,
                                                    mode = 'min')

    with strategy.scope():
        model = get_arc_model(n_classes=n_classes,
                              dense_layers=[512],
                              learning_rate=args.learning_rate)
        model.fit(train_dataset,
                  epochs=args.epochs,
                  validation_data=val_dataset,
                  callbacks=[tensorboard_callback,
                             lr_callback,
                             checkpoint],
                  verbose=2)
        model.evaluate(test_dataset, verbose=2)

