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
    python3 cnn4.py --mode train --data_dir /path/to/UCSDped1 --model_dir ./models
    python3 cnn4.py --mode eval --data_dir /path/to/UCSDped1 --model_dir ./models

"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'UCSD_Anomaly_Dataset.v1p2' / 'UCSDped1'

# Heavy ML imports are done lazily inside functions so `--help` works without
# large packages installed.


def list_image_files(folder: Path, exts=None) -> Iterable[Path]:
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    for p in sorted(folder.rglob('*')):
        if any(part.endswith('_gt') for part in p.parts):
            continue
        if p.suffix.lower() in exts:
            yield p


def resolve_split_root(data_dir: str, split_name: str) -> Path:
    """Resolve the directory that contains the actual UCSD Ped1 frames.

    Accepts either the dataset root (UCSDped1), the archive root
    (UCSD_Anomaly_Dataset.v1p2/UCSDped1), or the split folder itself
    (Train / Test / train / test).
    """
    base = Path(data_dir).expanduser().resolve()
    split_candidates = [split_name, split_name.lower(), split_name.upper()]
    dataset_roots = [base, base / 'UCSDped1', base / 'UCSD_Anomaly_Dataset.v1p2' / 'UCSDped1']

    # If the caller already passed the split folder, use it directly.
    if base.exists() and base.is_dir() and base.name.lower() == split_name.lower():
        return base

    # If the caller passed the dataset root, prefer a matching Train/Test child.
    for dataset_root in dataset_roots:
        for name in split_candidates:
            split_dir = dataset_root / name
            if split_dir.exists() and split_dir.is_dir():
                return split_dir

    # Fall back to any existing matching folder anywhere in the known roots.
    for dataset_root in dataset_roots:
        for name in split_candidates:
            split_dir = dataset_root / name
            if split_dir.exists() and split_dir.is_dir():
                return split_dir

    raise FileNotFoundError(f'Could not resolve {split_name} split under {data_dir}')


def load_frames_from_directory(
    directory: str,
    image_size: Tuple[int, int],
    strict: bool = True,
    return_paths: bool = False,
):
    """Load images recursively and return frames in [0, 1].

    When strict=False, unreadable files are skipped with a warning.
    If return_paths=True, returns a tuple: (frames, loaded_file_paths).
    """
    try:
        import cv2
    except Exception:
        cv2 = None
    try:
        from PIL import Image
    except Exception:
        Image = None

    root = Path(directory)
    files = list(list_image_files(root))
    if len(files) == 0:
        raise FileNotFoundError(f'No images found in {directory}')

    h, w = image_size
    loaded_images = []
    loaded_files = []

    def _load_with_pil(path: Path) -> np.ndarray:
        if Image is None:
            raise RuntimeError('PIL not available')
        img = Image.open(path).convert('L').resize((w, h))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_with_cv2(path: Path) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError('OpenCV not available')
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f'Failed to read image {path}')
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    for p in files:
        try:
            # Prefer PIL for TIFF files because OpenCV emits noisy decode warnings.
            if p.suffix.lower() in {'.tif', '.tiff'}:
                try:
                    img = _load_with_pil(p)
                except Exception:
                    img = _load_with_cv2(p)
            else:
                try:
                    img = _load_with_cv2(p)
                except Exception:
                    img = _load_with_pil(p)
        except Exception as exc:
            if strict:
                raise RuntimeError(f'Failed to read image {p}') from exc
            print(f'Warning: skipping unreadable image {p} ({exc})')
            continue

        loaded_images.append(img)
        loaded_files.append(p)

    if not loaded_images:
        raise RuntimeError(f'No readable images found in {directory}')

    out = np.zeros((len(loaded_images), h, w, 1), dtype=np.float32)
    for i, img in enumerate(loaded_images):
        out[i, :, :, 0] = img

    if return_paths:
        return out, loaded_files
    return out


def build_autoencoder(input_shape: Tuple[int, int, int]):
    """Builds a small convolutional autoencoder using tf.keras.

    Input: (H, W, C) returns compiled model (autoencoder) and encoder model.
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(128, activation='relu', name='latent')(x)

    # Decoder
    x = tf.keras.layers.Dense(int(np.prod(shape_before_flatten)), activation='relu')(latent)
    x = tf.keras.layers.Reshape(shape_before_flatten)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
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


def score_reconstruction_errors(
    frames: np.ndarray,
    reconstructions: np.ndarray,
    mode: str = 'mean',
    percentile: float = 95.0,
    patch_size: int = 16,
) -> np.ndarray:
    """Compute per-frame anomaly scores from reconstruction residuals.

    Modes:
    - mean: global frame MSE
    - p95: percentile of per-pixel squared error per frame
    - patchmax: max mean error over non-overlapping patches
    """
    err_map = np.mean((frames - reconstructions) ** 2, axis=-1)  # (N, H, W)

    if mode == 'mean':
        return np.mean(err_map, axis=(1, 2))

    if mode == 'p95':
        flat = err_map.reshape((err_map.shape[0], -1))
        return np.percentile(flat, percentile, axis=1)

    if mode == 'patchmax':
        ps = max(1, int(patch_size))
        n, h, w = err_map.shape
        h2 = (h // ps) * ps
        w2 = (w // ps) * ps
        if h2 == 0 or w2 == 0:
            return np.mean(err_map, axis=(1, 2))

        cropped = err_map[:, :h2, :w2]
        blocks = cropped.reshape(n, h2 // ps, ps, w2 // ps, ps)
        block_means = blocks.mean(axis=(2, 4))
        return block_means.max(axis=(1, 2))

    raise ValueError(f'Unsupported score_mode: {mode}')


def train(args):
    import tensorflow as tf

    train_dir = resolve_split_root(args.data_dir, 'Train')

    print(f'Loading training frames from {train_dir} ...')
    frames = load_frames_from_directory(str(train_dir), (args.height, args.width))
    if args.frame_stride > 1:
        frames = frames[::args.frame_stride]
        print(f'Using every {args.frame_stride}th training frame; reduced to', frames.shape)
    print('Loaded', frames.shape)

    autoencoder, encoder = build_autoencoder((args.height, args.width, 1))

    val_size = max(1, int(len(frames) * 0.1))
    train_frames = frames[:-val_size]
    val_frames = frames[-val_size:]

    train_ds = prepare_tf_dataset(train_frames, args.batch_size, shuffle=True, for_training=True)
    val_ds = prepare_tf_dataset(val_frames, args.batch_size, shuffle=False, for_training=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.model_dir, 'ae_best.keras'), save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ]

    os.makedirs(args.model_dir, exist_ok=True)

    history = autoencoder.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # save final model
    autoencoder.save(os.path.join(args.model_dir, 'ae_final.keras'))
    print('Training complete. Models saved to', args.model_dir)


def evaluate(args):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    test_dir = resolve_split_root(args.data_dir, 'Test')

    print(f'Loading test frames from {test_dir} ...')
    frames, frame_files = load_frames_from_directory(
        str(test_dir),
        (args.height, args.width),
        strict=False,
        return_paths=True,
    )
    print('Loaded', frames.shape)

    model_path = Path(args.model_dir) / 'ae_best.keras'
    if not model_path.exists():
        model_path = Path(args.model_dir) / 'ae_final.keras'
    if not model_path.exists():
        model_path = Path(args.model_dir) / 'ae_best.h5'
    if not model_path.exists():
        model_path = Path(args.model_dir) / 'ae_final.h5'
    if not model_path.exists():
        raise FileNotFoundError(f'No model found in {args.model_dir}')

    print('Loading model from', model_path)
    autoencoder = tf.keras.models.load_model(str(model_path))

    # Predict in batches
    ds = prepare_tf_dataset(frames, args.batch_size, shuffle=False, for_training=False)
    reconstructions = autoencoder.predict(ds)

    # Compute per-frame anomaly scores from reconstruction residuals.
    errors = score_reconstruction_errors(
        frames,
        reconstructions,
        mode=args.score_mode,
        percentile=args.score_percentile,
        patch_size=args.patch_size,
    )

    out_errors = Path(args.model_dir) / 'reconstruction_errors.npy'
    np.save(out_errors, errors)
    print('Saved reconstruction errors to', out_errors)
    print(
        f'Anomaly score mode={args.score_mode}, '
        f'percentile={args.score_percentile:.1f}, patch_size={args.patch_size}'
    )

    # Build evaluation labels from available Ped1 *_gt masks.
    try:
        import cv2
    except Exception:
        cv2 = None

    def _load_mask(mask_path: Path) -> np.ndarray:
        if cv2 is not None:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f'Failed to read mask {mask_path}')
            m = cv2.resize(m, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
            return m.astype(np.float32) / 255.0
        from PIL import Image
        m = Image.open(mask_path).convert('L').resize((args.width, args.height))
        return np.asarray(m, dtype=np.float32) / 255.0

    y_true_list = []
    gt_eval_indices = []
    for idx, frame_path in enumerate(frame_files):
        seq_dir = frame_path.parent
        gt_dir = seq_dir.parent / f'{seq_dir.name}_gt'
        if not gt_dir.exists():
            continue
        gt_mask = gt_dir / f'{frame_path.stem}.bmp'
        if not gt_mask.exists():
            continue
        mask = _load_mask(gt_mask)
        y_true_list.append(1 if np.any(mask > 0.0) else 0)
        gt_eval_indices.append(idx)

    y_true = np.asarray(y_true_list, dtype=np.int32) if y_true_list else None

    # Select threshold mode.
    if args.threshold_mode == 'std':
        threshold = float(errors.mean() + 2.0 * errors.std())
        threshold_desc = 'mean + 2*std'
    elif args.threshold_mode == 'p95':
        threshold = float(np.percentile(errors, 95))
        threshold_desc = '95th percentile'
    elif args.threshold_mode == 'p97':
        threshold = float(np.percentile(errors, 97))
        threshold_desc = '97th percentile'
    elif args.threshold_mode == 'p99':
        threshold = float(np.percentile(errors, 99))
        threshold_desc = '99th percentile'
    elif args.threshold_mode == 'f1':
        if y_true is None or y_true.size == 0:
            threshold = float(errors.mean() + 2.0 * errors.std())
            threshold_desc = 'mean + 2*std (fallback; no GT masks available for F1 search)'
        else:
            eval_errors = errors[np.asarray(gt_eval_indices, dtype=np.int32)]
            # Candidate thresholds sampled from the evaluable error range.
            lo, hi = float(eval_errors.min()), float(eval_errors.max())
            candidates = np.linspace(lo, hi, num=200)
            best_f1 = -1.0
            threshold = candidates[0]
            for t in candidates:
                yp = (eval_errors > t).astype(np.int32)
                tp = float(np.sum((y_true == 1) & (yp == 1)))
                fp = float(np.sum((y_true == 0) & (yp == 1)))
                fn = float(np.sum((y_true == 1) & (yp == 0)))
                precision_tmp = tp / (tp + fp) if (tp + fp) else 0.0
                recall_tmp = tp / (tp + fn) if (tp + fn) else 0.0
                f1_tmp = (2.0 * precision_tmp * recall_tmp / (precision_tmp + recall_tmp)) if (precision_tmp + recall_tmp) else 0.0
                if f1_tmp > best_f1:
                    best_f1 = f1_tmp
                    threshold = t
            threshold = float(threshold)
            threshold_desc = f'best F1 over GT-masked test frames (F1={best_f1:.4f})'
    else:
        raise ValueError(f'Unsupported threshold_mode: {args.threshold_mode}')

    y_pred = (errors > threshold).astype(np.int32)

    # Optional temporal smoothing per clip to reduce isolated false positives.
    if args.smooth_window > 1:
        clip_to_indices = {}
        for i, fp in enumerate(frame_files):
            clip = fp.parent.name
            clip_to_indices.setdefault(clip, []).append(i)

        half = args.smooth_window // 2
        y_smooth = y_pred.copy()
        for _, indices in clip_to_indices.items():
            clip_arr = y_pred[np.asarray(indices, dtype=np.int32)]
            clip_out = clip_arr.copy()
            for j in range(len(clip_arr)):
                lo = max(0, j - half)
                hi = min(len(clip_arr), j + half + 1)
                votes = int(np.sum(clip_arr[lo:hi]))
                clip_out[j] = 1 if votes >= ((hi - lo) // 2 + 1) else 0
            y_smooth[np.asarray(indices, dtype=np.int32)] = clip_out
        y_pred = y_smooth
        print(f'Applied temporal smoothing with window={args.smooth_window}')
    out_pred = Path(args.model_dir) / 'predicted_labels.npy'
    np.save(out_pred, y_pred)
    print('Saved predicted anomaly labels to', out_pred)
    print(f'Anomaly threshold ({threshold_desc}): {threshold:.8f}')

    if y_true is not None and y_true.size > 0:
        y_pred_eval = y_pred[np.asarray(gt_eval_indices, dtype=np.int32)]
        try:
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                precision_recall_fscore_support,
            )

            cm = confusion_matrix(y_true, y_pred_eval, labels=[0, 1])
            acc = float(accuracy_score(y_true, y_pred_eval))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred_eval,
                average='binary',
                zero_division=0,
            )
            report = classification_report(
                y_true,
                y_pred_eval,
                target_names=['normal', 'anomaly'],
                digits=4,
                zero_division=0,
            )
        except Exception:
            tn = int(np.sum((y_true == 0) & (y_pred_eval == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred_eval == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred_eval == 0)))
            tp = int(np.sum((y_true == 1) & (y_pred_eval == 1)))
            cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
            acc = float((tp + tn) / len(y_true))
            precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            report = (
                'scikit-learn unavailable: generated fallback metrics only.\n'
                f'accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}'
            )

        print(f'Evaluation used {len(y_true)} frames from test clips with GT masks.')
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
    else:
        print('No *_gt masks found in test set; skipped Accuracy/Precision/Recall/F1 and confusion matrix.')

    if args.plot:
        # plot histogram of errors
        plt.figure()
        plt.hist(errors, bins=100)
        plt.title('Frame reconstruction MSE')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.savefig(Path(args.model_dir) / 'error_hist.png')
        print('Saved error histogram to', Path(args.model_dir) / 'error_hist.png')

        if y_true is not None and y_true.size > 0:
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
    p.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR), help='Path to the UCSD Ped1 dataset root or split folder')
    p.add_argument('--model_dir', default='./models', help='Where to save/load models')
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--frame_stride', type=int, default=4, help='Use every Nth training frame to reduce training time')
    p.add_argument('--score_mode', choices=['mean', 'p95', 'patchmax'], default='mean', help='Scoring method for reconstruction residuals during eval')
    p.add_argument('--score_percentile', type=float, default=95.0, help='Percentile used when score_mode=p95')
    p.add_argument('--patch_size', type=int, default=16, help='Patch size used when score_mode=patchmax')
    p.add_argument('--threshold_mode', choices=['std', 'p95', 'p97', 'p99', 'f1'], default='std', help='Threshold strategy during evaluation')
    p.add_argument('--smooth_window', type=int, default=1, help='Temporal smoothing window size for eval predictions (odd integer; 1 disables)')
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
