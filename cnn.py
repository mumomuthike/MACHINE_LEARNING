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
import re
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
        # Skip common metadata/sidecar files from macOS archives.
        if p.name.startswith('._') or p.name == '.DS_Store':
            continue
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
    x = tf.keras.layers.Dense(int(np.prod(shape_before_flatten)), activation='relu')(latent)
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


def resolve_model_path(model_dir: Path) -> Path:
    candidates = [
        model_dir / 'ae_best.keras',
        model_dir / 'ae_final.keras',
        model_dir / 'ae_best.h5',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No model found in {model_dir}')


def load_autoencoder(model_dir: Path):
    import tensorflow as tf

    model_path = resolve_model_path(model_dir)
    print('Loading model from', model_path)
    autoencoder = tf.keras.models.load_model(str(model_path), compile=False)
    return autoencoder, model_path


def _train_single(frames_train: np.ndarray, frames_val: np.ndarray, args, checkpoint_path: str):
    """Train one autoencoder from scratch. Returns (model, best_val_loss, history)."""
    import tensorflow as tf

    autoencoder, _ = build_autoencoder((args.height, args.width, 1))

    ds_train = prepare_tf_dataset(frames_train, args.batch_size, shuffle=True, for_training=True)
    ds_val = prepare_tf_dataset(frames_val, args.batch_size, shuffle=False, for_training=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    history = autoencoder.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    best_val_loss = float(min(history.history['val_loss']))
    return autoencoder, best_val_loss, history


def train(args):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    train_dir = Path(args.data_dir) / 'Train'
    if not train_dir.exists():
        train_dir = Path(args.data_dir)

    print(f'Loading training frames from {train_dir} ...')
    frames = load_frames_from_directory(str(train_dir), (args.height, args.width))
    print('Loaded', frames.shape)

    os.makedirs(args.model_dir, exist_ok=True)
    k = args.folds
    print(f'\nStarting {k}-fold cross-validation ...')

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(frames))
    fold_size = len(frames) // k

    fold_val_losses: list[float] = []
    best_overall_loss = float('inf')
    best_fold = -1

    for fold in range(k):
        print(f'\n--- Fold {fold + 1}/{k} ---')
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([
            indices[: fold * fold_size],
            indices[(fold + 1) * fold_size :],
        ])
        frames_train = frames[train_idx]
        frames_val = frames[val_idx]

        checkpoint_path = os.path.join(args.model_dir, f'ae_fold_{fold + 1}.keras')
        _, val_loss, _ = _train_single(frames_train, frames_val, args, checkpoint_path)

        fold_val_losses.append(val_loss)
        print(f'Fold {fold + 1} best val_loss: {val_loss:.6f}')

        if val_loss < best_overall_loss:
            best_overall_loss = val_loss
            best_fold = fold + 1

    # Summary
    mean_loss = float(np.mean(fold_val_losses))
    std_loss = float(np.std(fold_val_losses))
    print(f'\nCross-validation complete.')
    print(f'  Val loss per fold: {[f"{v:.6f}" for v in fold_val_losses]}')
    print(f'  Mean: {mean_loss:.6f}  Std: {std_loss:.6f}')
    print(f'  Best fold: {best_fold} (val_loss={best_overall_loss:.6f})')

    # Promote best fold checkpoint as ae_best.keras
    import shutil
    best_fold_path = os.path.join(args.model_dir, f'ae_fold_{best_fold}.keras')
    best_path = os.path.join(args.model_dir, 'ae_best.keras')
    shutil.copy2(best_fold_path, best_path)
    print(f'Best fold model copied to {best_path}')

    # Clean up per-fold checkpoints
    for fold in range(k):
        fold_path = Path(args.model_dir) / f'ae_fold_{fold + 1}.keras'
        if fold_path.exists():
            fold_path.unlink()

    # Final model: retrain on all data
    print('\nRetraining final model on all data ...')
    final_checkpoint = os.path.join(args.model_dir, '_ae_final_ckpt.keras')
    autoencoder, _, _ = _train_single(frames, frames, args, final_checkpoint)
    autoencoder.save(os.path.join(args.model_dir, 'ae_final.keras'))
    Path(final_checkpoint).unlink(missing_ok=True)

    if args.plot:
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, k + 1), fold_val_losses, color='#0b5fff', edgecolor='black')
        plt.axhline(mean_loss, linestyle='--', color='#c62828', label=f'Mean = {mean_loss:.6f}')
        plt.xlabel('Fold')
        plt.ylabel('Best val MSE')
        plt.title(f'{k}-Fold Cross-Validation — Reconstruction MSE')
        plt.xticks(range(1, k + 1))
        plt.legend()
        plt.tight_layout()
        cv_plot_path = Path(args.model_dir) / 'cv_val_loss.png'
        plt.savefig(cv_plot_path, dpi=150)
        print(f'Saved CV plot to {cv_plot_path}')

    print('Training complete. Models saved to', args.model_dir)


def evaluate(args):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    test_dir = Path(args.data_dir) / 'Test'
    if not test_dir.exists():
        test_dir = Path(args.data_dir)

    print(f'Loading test frames from {test_dir} ...')

    seq_dirs = sorted(
        [
            d for d in test_dir.iterdir()
            if d.is_dir() and re.match(r'^Test\d{3}$', d.name)
        ],
        key=lambda p: p.name,
    ) if test_dir.exists() else []

    if seq_dirs:
        h, w = args.height, args.width
        frames_list: list[np.ndarray] = []
        labels_list: list[int] = []
        has_any_gt = False

        try:
            import cv2
        except Exception:
            cv2 = None

        def _load_image(path: Path, interpolation):
            if cv2 is not None:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f'Failed to read image {path}')
                img = cv2.resize(img, (w, h), interpolation=interpolation)
                return img.astype(np.float32) / 255.0

            from PIL import Image

            pil_img = Image.open(path).convert('L').resize((w, h))
            return np.asarray(pil_img, dtype=np.float32) / 255.0

        for seq_dir in seq_dirs:
            frame_paths = list(list_image_files(seq_dir))
            gt_dir = seq_dir.parent / f'{seq_dir.name}_gt'
            gt_paths = list(list_image_files(gt_dir)) if gt_dir.exists() else []

            if gt_dir.exists():
                has_any_gt = True
                if len(gt_paths) != len(frame_paths):
                    raise RuntimeError(
                        f'GT/frame count mismatch in {seq_dir.name}: '
                        f'{len(gt_paths)} gt vs {len(frame_paths)} frames'
                    )

            for i, frame_path in enumerate(frame_paths):
                frame = _load_image(frame_path, interpolation=3 if cv2 is None else cv2.INTER_AREA)
                frames_list.append(frame)

                if gt_dir.exists():
                    gt = _load_image(gt_paths[i], interpolation=0 if cv2 is None else cv2.INTER_NEAREST)
                    labels_list.append(1 if np.any(gt > 0.0) else 0)
                else:
                    labels_list.append(0)

        frames = np.asarray(frames_list, dtype=np.float32)[..., np.newaxis]
        y_true = np.asarray(labels_list, dtype=np.int32) if has_any_gt else None
    else:
        # Fallback for flat test folder layouts that contain only frames.
        frames = load_frames_from_directory(str(test_dir), (args.height, args.width))
        y_true = None

    print('Loaded', frames.shape)

    autoencoder, model_path = load_autoencoder(Path(args.model_dir))

    # Predict in batches
    ds = prepare_tf_dataset(frames, args.batch_size, shuffle=False, for_training=False)
    reconstructions = autoencoder.predict(ds)

    # Flatten per-frame MSE
    errors = np.mean((frames - reconstructions) ** 2, axis=(1, 2, 3))
    threshold = float(errors.mean() + 2.0 * errors.std())
    y_pred = (errors > threshold).astype(np.int32)

    out_errors = Path(args.model_dir) / 'reconstruction_errors.npy'
    np.save(out_errors, errors)
    print('Saved reconstruction errors to', out_errors)

    out_pred = Path(args.model_dir) / 'predicted_labels.npy'
    np.save(out_pred, y_pred)
    print('Saved predicted anomaly labels to', out_pred)

    print(f'Anomaly threshold (mean + 2*std): {threshold:.8f}')

    if y_true is not None:
        try:
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                precision_recall_fscore_support,
            )

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = float(accuracy_score(y_true, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='binary',
                zero_division=0,
            )
            report = classification_report(
                y_true,
                y_pred,
                target_names=['normal', 'anomaly'],
                digits=4,
                zero_division=0,
            )
        except Exception:
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
            acc = float((tp + tn) / len(y_true))
            precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            report = (
                'scikit-learn unavailable: generated fallback metrics only.\n'
                f'accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}'
            )

        print('Confusion matrix [[TN, FP], [FN, TP]]:')
        print(cm)
        print(f'Accuracy: {acc:.4f}')
        print(f'Precision: {float(precision):.4f}')
        print(f'Recall: {float(recall):.4f}')
        print(f'F1: {float(f1):.4f}')

        out_cm = Path(args.model_dir) / 'confusion_matrix.npy'
        np.save(out_cm, cm)
        print('Saved confusion matrix to', out_cm)

        out_report = Path(args.model_dir) / 'classification_report.txt'
        with open(out_report, 'w', encoding='utf-8') as f:
            f.write('Confusion matrix [[TN, FP], [FN, TP]]\n')
            f.write(str(cm))
            f.write('\n\n')
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'Precision: {float(precision):.4f}\n')
            f.write(f'Recall: {float(recall):.4f}\n')
            f.write(f'F1: {float(f1):.4f}\n\n')
            f.write('Classification report:\n')
            f.write(report)
            f.write('\n')
        print('Saved classification report to', out_report)

    if args.plot:
        # plot histogram of errors
        plt.figure()
        plt.hist(errors, bins=100)
        plt.title('Frame reconstruction MSE')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.savefig(Path(args.model_dir) / 'error_hist.png')
        print('Saved error histogram to', Path(args.model_dir) / 'error_hist.png')

        if y_true is not None:
            cm_plot_path = Path(args.model_dir) / 'confusion_matrix.png'
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.xticks([0, 1], ['normal', 'anomaly'])
            plt.yticks([0, 1], ['normal', 'anomaly'])

            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

            plt.tight_layout()
            plt.savefig(cm_plot_path)
            print('Saved confusion matrix plot to', cm_plot_path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Conv Autoencoder for UCSD Ped1 frames')
    p.add_argument('--data_dir', required=True, help='Path to dataset (Train/Test subfolders or flat frames)')
    p.add_argument('--model_dir', default='./models', help='Where to save/load models')
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--width', type=int, default=160, help='Frame width (pixels)')
    p.add_argument('--height', type=int, default=240, help='Frame height (pixels)')
    p.add_argument('--plot', action='store_true', help='Produce simple plots during evaluation/training')
    p.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation during training (default: 5)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)


if __name__ == '__main__':
    main()
