import abc

import tensorflow as tf
import argparse
import os
import json
import pandas as pd

import math
from typing import List


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


class BaseDataLoader(abc.ABC):
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
            base_path="/opt/ml/input/data/train/datasets/train_images_rescaled_partial",
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

    LOG_DIR = "/opt/ml/output/tensorboard"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, histogram_freq=1)

    data_path = '/opt/ml/input/data/train'
    train_path = os.path.join(data_path, 'train_images_rescaled_partial')
    test_path = os.path.join(data_path, 'clown_fiesta')
    train_csv_path = os.path.join(data_path, 'train.csv')
    test_csv_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(train_csv_path)

    train_image_paths = train_df['image'].tolist()
    train_image_labels = train_df['label_group'].tolist()
    arc_face_data_loader = ArcFaceDataLoader(
        base_path=train_path,
        labels=train_image_labels,
        filenames=train_image_paths,
        target_width=300,
        target_height=300,
        batch_size=32,
        shuffle_buffer_size=50,
        data_augmentation=True)

    test_image_paths = train_df['image'].tolist()
    test_image_labels = train_df['label_group'].tolist()
    test_data_loader = ArcFaceDataLoader(
        base_path=test_path,
        labels=test_image_labels,
        filenames=test_image_paths,
        target_width=300,
        target_height=300,
        batch_size=32,
        shuffle_buffer_size=50,
        data_augmentation=True)

    train_dataset = arc_face_data_loader.get_dataset()
    test_dataset = test_data_loader.get_dataset()

    n_classes = int(max(max(train_image_labels), max(test_image_labels)) + 1)

    model = get_arc_model(n_classes=n_classes, dense_layers=[512])
    model.fit(train_dataset, epochs=args.epochs, callbacks=tensorboard_callback)
    model.evaluate(test_dataset)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
