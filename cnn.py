#!/usr/bin/env python3
from __future__ import annotations
"""
Convolutional Autoencoder for UCSD Ped1 (frame-level anomaly detection)

This script provides a minimal, runnable convolutional autoencoder and
dataset helpers geared to the UCSD Ped1 dataset (frames saved per video).

Assumptions:
- The dataset root contains subfolders for training and testing, e.g.
  dataset_root/Train/Train001/ (image files) and dataset_root/Test/Test001/
- Frames are individual image files (png/jpg/tif). If you only have videos,
  extract frames first (ffmpeg can do this).

Usage examples (see README.md):
  python3 cnn.py --mode train --data_dir /path/to/UCSDped1 --model_dir ./models
  python3 cnn.py --mode eval --data_dir /path/to/UCSDped1 --model_dir ./models

"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np

# Heavy ML imports are done lazily inside functions so `--help` works without
# large packages installed.


def list_image_files(folder: Path, exts=None) -> Iterable[Path]:
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    for p in sorted(folder.rglob('*')):
        if p.suffix.lower() in exts:
            yield p


def load_frames_from_directory(directory: str, image_size: Tuple[int, int]) -> np.ndarray:
    """Load all images under `directory` (recursively), return as float32 numpy array.

    Returns array of shape (N, H, W, 1) with values in [0, 1].
    """
    try:
        import cv2
    except Exception:
        cv2 = None

    root = Path(directory)
    files = list(list_image_files(root))
    if len(files) == 0:
        raise FileNotFoundError(f'No images found in {directory}')

    h, w = image_size
    out = np.zeros((len(files), h, w, 1), dtype=np.float32)

    for i, p in enumerate(files):
        if cv2 is not None:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f'Failed to read image {p}')
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
        else:
            # fallback to PIL (slower, but avoids extra deps)
            from PIL import Image

            img = Image.open(p).convert('L').resize((w, h))
            img = np.asarray(img, dtype=np.float32) / 255.0

        out[i, :, :, 0] = img

    return out


def build_autoencoder(input_shape: Tuple[int, int, int]):
    """Builds a small convolutional autoencoder using tf.keras.

    Input: (H, W, C) returns compiled model (autoencoder) and encoder model.
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(256, activation='relu', name='latent')(x)

    # Decoder
    x = tf.keras.layers.Dense(np.prod(shape_before_flatten), activation='relu')(latent)
    x = tf.keras.layers.Reshape(shape_before_flatten)(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2D(input_shape[2], 3, padding='same', activation='sigmoid')(x)

    autoencoder = tf.keras.Model(inputs, outputs, name='conv_autoencoder')
    encoder = tf.keras.Model(inputs, latent, name='encoder')

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    return autoencoder, encoder


def prepare_tf_dataset(frames: np.ndarray, batch_size: int, shuffle: bool = True, for_training: bool = True):
    """Return a tf.data.Dataset.

    If for_training is True the dataset yields (x, x) pairs so the
    autoencoder receives inputs and targets. Otherwise it yields inputs only
    (for prediction/evaluation).
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(frames.astype(np.float32))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(frames))

    if for_training:
        # map to (input, target) pairs for autoencoder training
        ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train(args):
    import tensorflow as tf

    train_dir = Path(args.data_dir) / 'Train'
    if not train_dir.exists():
        # allow direct directory
        train_dir = Path(args.data_dir)

    print(f'Loading training frames from {train_dir} ...')
    frames = load_frames_from_directory(str(train_dir), (args.height, args.width))
    print('Loaded', frames.shape)

    autoencoder, encoder = build_autoencoder((args.height, args.width, 1))

    ds = prepare_tf_dataset(frames, args.batch_size, shuffle=True, for_training=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.model_dir, 'ae_best.h5'), save_best_only=True, monitor='loss'),
        # monitor training loss because no validation set is provided by default
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, verbose=1),
    ]

    os.makedirs(args.model_dir, exist_ok=True)

    history = autoencoder.fit(ds, epochs=args.epochs, callbacks=callbacks)

    # save final model
    autoencoder.save(os.path.join(args.model_dir, 'ae_final'))
    print('Training complete. Models saved to', args.model_dir)


def evaluate(args):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    test_dir = Path(args.data_dir) / 'Test'
    if not test_dir.exists():
        test_dir = Path(args.data_dir)

    print(f'Loading test frames from {test_dir} ...')
    frames = load_frames_from_directory(str(test_dir), (args.height, args.width))
    print('Loaded', frames.shape)

    model_path = Path(args.model_dir) / 'ae_best.h5'
    if not model_path.exists():
        model_path = Path(args.model_dir) / 'ae_final'
    if not model_path.exists():
        raise FileNotFoundError(f'No model found in {args.model_dir}')

    print('Loading model from', model_path)
    autoencoder = tf.keras.models.load_model(str(model_path))

    # Predict in batches
    ds = prepare_tf_dataset(frames, args.batch_size, shuffle=False, for_training=False)
    reconstructions = autoencoder.predict(ds)

    # Flatten per-frame MSE
    errors = np.mean((frames - reconstructions) ** 2, axis=(1, 2, 3))

    out_errors = Path(args.model_dir) / 'reconstruction_errors.npy'
    np.save(out_errors, errors)
    print('Saved reconstruction errors to', out_errors)

    if args.plot:
        # plot histogram of errors
        plt.figure()
        plt.hist(errors, bins=100)
        plt.title('Frame reconstruction MSE')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.savefig(Path(args.model_dir) / 'error_hist.png')
        print('Saved error histogram to', Path(args.model_dir) / 'error_hist.png')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Conv Autoencoder for UCSD Ped1 frames')
    p.add_argument('--data_dir', required=True, help='Path to dataset (Train/Test subfolders or flat frames)')
    p.add_argument('--model_dir', default='./models', help='Where to save/load models')
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--width', type=int, default=160, help='Frame width (pixels)')
    p.add_argument('--height', type=int, default=240, help='Frame height (pixels)')
    p.add_argument('--plot', action='store_true', help='Produce simple plots during evaluation')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)


if __name__ == '__main__':
    main()
