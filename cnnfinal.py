from __future__ import annotations
"""
Supervised CNN Frame Classifier for UCSD Ped1 Anomaly Detection

This version is for the professor-requested supervised/semi-supervised pivot:
- UCSDped1/Train frames are labeled normal = 0
- UCSDped1/Test frames are labeled using *_gt ground-truth masks:
    0 = normal frame
    1 = anomalous frame
- The combined labeled dataset is split into train/validation/held-out test.
- The CNN predicts frame-level anomaly probability using binary crossentropy.

Example train:
  python3 cnnfinal.py --mode train --data_dir ./UCSDped1 --model_dir ./models_supervised --epochs 12 --frame_stride 4 --plot

Example eval:
  python3 cnnfinal.py --mode eval --data_dir ./UCSDped1 --model_dir ./models_supervised --plot
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


# -----------------------------
# File / image helpers
# -----------------------------

def list_image_files(folder: Path, exts=None) -> list[Path]:
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for p in sorted(folder.rglob('*')):
        if p.name.startswith('._') or p.name == '.DS_Store':
            continue
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def list_sequence_dirs(root: Path, prefix: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(rf'^{prefix}\d{{3}}$', d.name)],
        key=lambda p: p.name,
    )


def load_gray_image(path: Path, image_size: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Returns grayscale float32 image in [0,1], shape (H,W). image_size=(H,W)."""
    h, w = image_size
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f'Failed to read image {path}')
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        img = cv2.resize(img, (w, h), interpolation=interp)
        return img.astype(np.float32) / 255.0
    except Exception:
        from PIL import Image
        resample = Image.NEAREST if nearest else Image.BILINEAR
        img = Image.open(path).convert('L').resize((w, h), resample=resample)
        return np.asarray(img, dtype=np.float32) / 255.0


# -----------------------------
# Dataset construction
# -----------------------------

def load_train_normal_frames(train_root: Path, image_size: Tuple[int, int], frame_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Load UCSDped1/Train frames, label all as normal=0."""
    seq_dirs = list_sequence_dirs(train_root, 'Train')
    if not seq_dirs:
        # fallback: recursively load image files but skip *_gt just in case
        frame_paths = [p for p in list_image_files(train_root) if not any(part.endswith('_gt') for part in p.parts)]
        frame_paths = frame_paths[::frame_stride]
    else:
        frame_paths = []
        for seq in seq_dirs:
            frame_paths.extend(list_image_files(seq)[::frame_stride])

    if not frame_paths:
        raise FileNotFoundError(f'No Train frames found in {train_root}')

    frames = [load_gray_image(p, image_size)[..., np.newaxis] for p in frame_paths]
    x = np.asarray(frames, dtype=np.float32)
    y = np.zeros((len(x),), dtype=np.int32)
    return x, y


def load_test_labeled_frames(test_root: Path, image_size: Tuple[int, int], frame_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Load UCSDped1/Test frames and label using Test###_gt masks when present."""
    seq_dirs = list_sequence_dirs(test_root, 'Test')
    if not seq_dirs:
        raise FileNotFoundError(f'No Test### clip folders found in {test_root}')

    all_frames: list[np.ndarray] = []
    all_labels: list[int] = []

    for seq_dir in seq_dirs:
        frame_paths = list_image_files(seq_dir)[::frame_stride]
        if not frame_paths:
            continue

        gt_dir = seq_dir.parent / f'{seq_dir.name}_gt'
        gt_paths = list_image_files(gt_dir)[::frame_stride] if gt_dir.exists() else []

        # If a GT folder exists, use it. If not, label as normal because no annotation is available.
        if gt_dir.exists() and len(gt_paths) != len(frame_paths):
            raise RuntimeError(
                f'GT/frame count mismatch for {seq_dir.name}: {len(gt_paths)} masks vs {len(frame_paths)} frames'
            )

        for i, frame_path in enumerate(frame_paths):
            frame = load_gray_image(frame_path, image_size)[..., np.newaxis]
            all_frames.append(frame)

            if gt_dir.exists():
                mask = load_gray_image(gt_paths[i], image_size, nearest=True)
                all_labels.append(1 if np.any(mask > 0.0) else 0)
            else:
                all_labels.append(0)

    if not all_frames:
        raise RuntimeError(f'Loaded 0 Test frames from {test_root}')

    return np.asarray(all_frames, dtype=np.float32), np.asarray(all_labels, dtype=np.int32)


def make_labeled_dataset(args) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(args.data_dir)
    train_root = data_dir / 'Train'
    test_root = data_dir / 'Test'

    if not train_root.exists():
        raise FileNotFoundError(f'Missing Train directory: {train_root}')
    if not test_root.exists():
        raise FileNotFoundError(f'Missing Test directory: {test_root}')

    image_size = (args.height, args.width)

    print('Loading normal Train frames...')
    x_train_normal, y_train_normal = load_train_normal_frames(train_root, image_size, args.frame_stride)
    print(f'  Train normal frames: {len(x_train_normal)}')

    print('Loading labeled Test frames using ground-truth masks...')
    x_test_labeled, y_test_labeled = load_test_labeled_frames(test_root, image_size, args.frame_stride)
    print(f'  Test labeled frames: {len(x_test_labeled)}')
    print(f'    normal={int(np.sum(y_test_labeled == 0))}, anomaly={int(np.sum(y_test_labeled == 1))}')

    x = np.concatenate([x_train_normal, x_test_labeled], axis=0)
    y = np.concatenate([y_train_normal, y_test_labeled], axis=0)

    print('\nCombined labeled dataset:')
    print(f'  total={len(x)}, normal={int(np.sum(y == 0))}, anomaly={int(np.sum(y == 1))}')

    return x, y


def split_dataset(x: np.ndarray, y: np.ndarray, seed: int = 42):
    """Stratified 70/15/15 split."""
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for label in [0, 1]:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


# -----------------------------
# Model
# -----------------------------

def build_cnn_classifier(input_shape: Tuple[int, int, int]):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs, name='supervised_ped1_cnn')

    try:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='roc_auc'),
        ],
    )
    return model


def make_tf_dataset(x, y, batch_size: int, shuffle: bool):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# -----------------------------
# Train / eval
# -----------------------------

def train(args):
    import tensorflow as tf
    import matplotlib.pyplot as plt

    os.makedirs(args.model_dir, exist_ok=True)

    x, y = make_labeled_dataset(args)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y, seed=args.seed)

    print('\nSplit sizes:')
    print(f'  train={len(x_train)} normal={int(np.sum(y_train == 0))} anomaly={int(np.sum(y_train == 1))}')
    print(f'  val={len(x_val)} normal={int(np.sum(y_val == 0))} anomaly={int(np.sum(y_val == 1))}')
    print(f'  heldout_test={len(x_test)} normal={int(np.sum(y_test == 0))} anomaly={int(np.sum(y_test == 1))}')

    # Save heldout split for repeatable eval
    np.save(Path(args.model_dir) / 'x_heldout_test.npy', x_test)
    np.save(Path(args.model_dir) / 'y_heldout_test.npy', y_test)

    ds_train = make_tf_dataset(x_train, y_train, args.batch_size, shuffle=True)
    ds_val = make_tf_dataset(x_val, y_val, args.batch_size, shuffle=False)

    model = build_cnn_classifier((args.height, args.width, 1))
    model.summary()

    # Class weights help because anomaly frames are usually fewer than normal frames.
    n_normal = np.sum(y_train == 0)
    n_anomaly = np.sum(y_train == 1)
    total = len(y_train)
    class_weight = {
        0: float(total / (2.0 * max(n_normal, 1))),
        1: float(total / (2.0 * max(n_anomaly, 1))),
    }
    print('Class weights:', class_weight)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(Path(args.model_dir) / 'cnn_best.keras'),
            monitor='val_roc_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_roc_auc',
            mode='max',
            patience=args.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(Path(args.model_dir) / 'training_log.csv')),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(Path(args.model_dir) / 'cnn_final.keras')

    with open(Path(args.model_dir) / 'split_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'train_total': int(len(y_train)),
            'train_normal': int(np.sum(y_train == 0)),
            'train_anomaly': int(np.sum(y_train == 1)),
            'val_total': int(len(y_val)),
            'val_normal': int(np.sum(y_val == 0)),
            'val_anomaly': int(np.sum(y_val == 1)),
            'heldout_test_total': int(len(y_test)),
            'heldout_test_normal': int(np.sum(y_test == 0)),
            'heldout_test_anomaly': int(np.sum(y_test == 1)),
            'frame_stride': args.frame_stride,
            'height': args.height,
            'width': args.width,
        }, f, indent=2)

    if args.plot:
        plt.figure()
        plt.plot(history.history.get('loss', []), label='train loss')
        plt.plot(history.history.get('val_loss', []), label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Crossentropy')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.model_dir) / 'training_loss.png', dpi=150)
        print('Saved plot:', Path(args.model_dir) / 'training_loss.png')

    print('\nTraining complete. Saved model to', args.model_dir)


def find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def evaluate(args):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_curve,
    )

    model_path = Path(args.model_dir) / 'cnn_best.keras'
    if not model_path.exists():
        model_path = Path(args.model_dir) / 'cnn_final.keras'
    if not model_path.exists():
        raise FileNotFoundError(f'No trained CNN model found in {args.model_dir}')

    x_path = Path(args.model_dir) / 'x_heldout_test.npy'
    y_path = Path(args.model_dir) / 'y_heldout_test.npy'

    if x_path.exists() and y_path.exists():
        print('Loading saved held-out test split...')
        x_test = np.load(x_path)
        y_test = np.load(y_path)
    else:
        print('No saved held-out test split found. Rebuilding dataset and split...')
        x, y = make_labeled_dataset(args)
        _, _, (x_test, y_test) = split_dataset(x, y, seed=args.seed)

    print(f'Evaluating on held-out test: total={len(y_test)}, normal={int(np.sum(y_test == 0))}, anomaly={int(np.sum(y_test == 1))}')

    model = tf.keras.models.load_model(model_path)
    scores = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)

    if args.threshold_mode == 'f1':
        threshold, best_f1 = find_best_f1_threshold(y_test, scores)
        print(f'Using F1-optimized threshold: {threshold:.6f} (search F1={best_f1:.4f})')
    else:
        threshold = args.threshold
        print(f'Using fixed threshold: {threshold:.6f}')

    y_pred = (scores >= threshold).astype(np.int32)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(pr_recall, pr_precision)

    print('\nConfusion Matrix [[TN, FP], [FN, TP]]')
    print(cm)
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1:        {f1:.4f}')
    print(f'ROC AUC:   {roc_auc:.4f}')
    print(f'PR AUC:    {pr_auc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['normal', 'anomaly'], digits=4, zero_division=0))

    results = {
        'threshold': float(threshold),
        'threshold_mode': args.threshold_mode,
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'test_total': int(len(y_test)),
        'test_normal': int(np.sum(y_test == 0)),
        'test_anomaly': int(np.sum(y_test == 1)),
    }
    out_json = Path(args.model_dir) / f'evaluation_supervised_{args.threshold_mode}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Saved results:', out_json)

    np.save(Path(args.model_dir) / 'supervised_scores.npy', scores)
    np.save(Path(args.model_dir) / 'supervised_y_true.npy', y_test)
    np.save(Path(args.model_dir) / 'supervised_y_pred.npy', y_pred)

    if args.plot:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks([0, 1], ['normal', 'anomaly'])
        plt.yticks([0, 1], ['normal', 'anomaly'])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.tight_layout()
        plt.savefig(Path(args.model_dir) / 'supervised_confusion_matrix.png', dpi=150)
        print('Saved plot:', Path(args.model_dir) / 'supervised_confusion_matrix.png')

        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.model_dir) / 'supervised_roc_curve.png', dpi=150)
        print('Saved plot:', Path(args.model_dir) / 'supervised_roc_curve.png')

        plt.figure()
        plt.plot(pr_recall, pr_precision, label=f'PR AUC={pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.model_dir) / 'supervised_precision_recall_curve.png', dpi=150)
        print('Saved plot:', Path(args.model_dir) / 'supervised_precision_recall_curve.png')

        plt.figure()
        plt.hist(scores[y_test == 0], bins=40, alpha=0.6, label='normal')
        plt.hist(scores[y_test == 1], bins=40, alpha=0.6, label='anomaly')
        plt.axvline(threshold, linestyle='--', label=f'threshold={threshold:.3f}')
        plt.xlabel('Predicted anomaly probability')
        plt.ylabel('Count')
        plt.title('Prediction Score Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.model_dir) / 'supervised_score_distribution.png', dpi=150)
        print('Saved plot:', Path(args.model_dir) / 'supervised_score_distribution.png')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Supervised CNN frame classifier for UCSD Ped1 anomaly detection')
    p.add_argument('--mode', choices=['train', 'eval'], required=True)
    p.add_argument('--data_dir', required=True, help='Path to UCSDped1 folder containing Train/ and Test/')
    p.add_argument('--model_dir', default='./models_supervised')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--width', type=int, default=160)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--frame_stride', type=int, default=4, help='Use every Nth frame to speed training')
    p.add_argument('--early_stop_patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--threshold_mode', choices=['fixed', 'f1'], default='fixed')
    p.add_argument('--threshold', type=float, default=0.5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
