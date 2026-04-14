from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def list_image_files(folder: Path, exts=None) -> Iterable[Path]:
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    for p in sorted(folder.rglob('*')):
        if p.name.startswith('._') or p.name == '.DS_Store':
            continue
        if p.suffix.lower() in exts:
            yield p


def load_frames_from_directory(directory: str, image_size: Tuple[int, int]) -> np.ndarray:
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
            from PIL import Image
            img = Image.open(p).convert('L').resize((w, h))
            img = np.asarray(img, dtype=np.float32) / 255.0

        out[i, :, :, 0] = img

    return out


def load_test_frames_and_labels(data_dir: str, image_size: Tuple[int, int]):
    test_dir = Path(data_dir) / 'Test'
    if not test_dir.exists():
        test_dir = Path(data_dir)

    seq_dirs = sorted(
        [
            d for d in test_dir.iterdir()
            if d.is_dir() and re.match(r'^Test\d{3}$', d.name)
        ],
        key=lambda p: p.name,
    ) if test_dir.exists() else []

    if not seq_dirs:
        frames = load_frames_from_directory(str(test_dir), image_size)
        return frames, None

    h, w = image_size
    frames_list = []
    labels_list = []
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
        img = Image.open(path).convert('L').resize((w, h))
        return np.asarray(img, dtype=np.float32) / 255.0

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
    return frames, y_true


def mean_frame_baseline(train_frames: np.ndarray, test_frames: np.ndarray):
    mean_frame = np.mean(train_frames, axis=0)
    errors = np.mean((test_frames - mean_frame) ** 2, axis=(1, 2, 3))
    return mean_frame, errors


def choose_threshold(errors: np.ndarray, mode: str, percentile: float):
    if mode == 'percentile':
        return float(np.percentile(errors, percentile))
    return float(errors.mean() + 2.0 * errors.std())


def evaluate_baseline(args):
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

    train_dir = Path(args.data_dir) / 'Train'
    if not train_dir.exists():
        train_dir = Path(args.data_dir)

    print(f'Loading training frames from {train_dir} ...')
    train_frames = load_frames_from_directory(str(train_dir), (args.height, args.width))
    print('Loaded training frames:', train_frames.shape)

    print(f'Loading test frames from {Path(args.data_dir) / "Test"} ...')
    test_frames, y_true = load_test_frames_and_labels(args.data_dir, (args.height, args.width))
    print('Loaded test frames:', test_frames.shape)

    mean_frame, errors = mean_frame_baseline(train_frames, test_frames)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'baseline_mean_frame.npy', mean_frame)
    np.save(output_dir / 'baseline_errors.npy', errors)

    print('Error percentiles:')
    for p in [80, 85, 90, 92, 95, 97, 99]:
        print(f'  {p}%: {np.percentile(errors, p):.6f}')

    threshold = choose_threshold(errors, args.threshold_mode, args.percentile)
    y_pred = (errors > threshold).astype(np.int32)

    np.save(output_dir / 'baseline_predicted_labels.npy', y_pred)
    np.save(output_dir / 'baseline_threshold.npy', np.array([threshold], dtype=np.float32))

    print(f'\nThreshold: {threshold:.8f}')

    if y_true is not None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        report = classification_report(
            y_true,
            y_pred,
            target_names=['normal', 'anomaly'],
            digits=4,
            zero_division=0,
        )

        fpr, tpr, _ = roc_curve(y_true, errors)
        roc_auc = float(auc(fpr, tpr))

        pr_precision, pr_recall, _ = precision_recall_curve(y_true, errors)
        pr_auc = float(auc(pr_recall, pr_precision))

        print('\nBaseline results')
        print('Confusion matrix [[TN, FP], [FN, TP]]')
        print(cm)
        print(f'\nAccuracy: {acc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')
        print(f'PR AUC: {pr_auc:.4f}')
        print('\nClassification report:')
        print(report)

        with open(output_dir / 'baseline_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write('Baseline results\n')
            f.write('Confusion matrix [[TN, FP], [FN, TP]]\n')
            f.write(str(cm))
            f.write('\n\n')
            f.write(f'Accuracy: {acc:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'F1: {f1:.4f}\n')
            f.write(f'Threshold: {threshold:.8f}\n')
            f.write(f'ROC AUC: {roc_auc:.4f}\n')
            f.write(f'PR AUC: {pr_auc:.4f}\n\n')
            f.write('Classification report:\n')
            f.write(report)
            f.write('\n')

        if args.plot:
            plt.figure()
            plt.hist(errors, bins=100)
            plt.axvline(threshold, linestyle='--', label=f'Threshold = {threshold:.6f}')
            plt.title('Baseline Frame Error Histogram')
            plt.xlabel('MSE from Mean Training Frame')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'baseline_error_hist.png')

            plt.figure(figsize=(5, 4))
            plt.imshow(cm, cmap='Blues')
            plt.title('Baseline Confusion Matrix')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.xticks([0, 1], ['normal', 'anomaly'])
            plt.yticks([0, 1], ['normal', 'anomaly'])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            plt.tight_layout()
            plt.savefig(output_dir / 'baseline_confusion_matrix.png')

            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Baseline ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'baseline_roc_curve.png')

            plt.figure()
            plt.plot(pr_recall, pr_precision, label=f'PR AUC = {pr_auc:.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Baseline Precision-Recall Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'baseline_pr_curve.png')

    else:
        print('No ground-truth labels found. Saved only errors, predictions, and threshold.')


def parse_args():
    p = argparse.ArgumentParser(description='Mean-frame baseline for UCSD Ped1 anomaly detection')
    p.add_argument('--data_dir', required=True, help='Path to dataset root')
    p.add_argument('--output_dir', default='./baseline_results', help='Directory for baseline outputs')
    p.add_argument('--width', type=int, default=160)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--plot', action='store_true', help='Save evaluation plots')
    p.add_argument(
        '--threshold_mode',
        choices=['std', 'percentile'],
        default='percentile',
        help='std = mean+2*std, percentile = np.percentile(errors, percentile)',
    )
    p.add_argument(
        '--percentile',
        type=float,
        default=95,
        help='Percentile used when threshold_mode=percentile',
    )
    return p.parse_args()


def main():
    args = parse_args()
    evaluate_baseline(args)


if __name__ == '__main__':
    main()