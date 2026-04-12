import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / 'models' / 'reconstruction_errors.npy'


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Visualize reconstruction errors saved by cnn.py eval mode.'
	)
	parser.add_argument(
		'--input',
		type=Path,
		default=DEFAULT_INPUT,
		help='Path to a .npy file containing reconstruction errors.',
	)
	parser.add_argument(
		'--save',
		type=Path,
		default=None,
		help='Optional output image path. If omitted, the plot is only shown on screen.',
	)
	parser.add_argument(
		'--no-show',
		action='store_true',
		help='Skip opening the interactive window. Useful when only saving the figure.',
	)
	return parser.parse_args()


def load_error_array(path: Path) -> np.ndarray:
	if not path.exists():
		raise FileNotFoundError(
			f'Could not find {path}. Run cnn.py in eval mode first to generate reconstruction_errors.npy.'
		)

	data = np.load(path)
	data = np.asarray(data).reshape(-1)
	if data.size == 0:
		raise ValueError(f'{path} is empty.')
	return data


def build_figure(errors: np.ndarray, source: Path) -> plt.Figure:
	threshold = float(errors.mean() + 2 * errors.std())

	fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
	fig.suptitle(f'Reconstruction Error Visualization\n{source.name}', fontsize=14)

	frame_axis = np.arange(errors.size)
	axes[0].plot(frame_axis, errors, color='#0b5fff', linewidth=1.5)
	axes[0].axhline(
		threshold,
		color='#c62828',
		linestyle='--',
		linewidth=1.2,
		label=f'Mean + 2 Std = {threshold:.6f}',
	)
	axes[0].fill_between(frame_axis, errors, color='#0b5fff', alpha=0.12)
	axes[0].set_title('Per-frame reconstruction error')
	axes[0].set_xlabel('Frame index')
	axes[0].set_ylabel('MSE')
	axes[0].grid(alpha=0.25)
	axes[0].legend()

	axes[1].hist(errors, bins=min(50, max(10, errors.size // 4)), color='#12a57a', edgecolor='black')
	axes[1].axvline(errors.mean(), color='#7b1fa2', linestyle='-', linewidth=1.2, label='Mean')
	axes[1].axvline(threshold, color='#c62828', linestyle='--', linewidth=1.2, label='Mean + 2 Std')
	axes[1].set_title('Distribution of reconstruction error')
	axes[1].set_xlabel('MSE')
	axes[1].set_ylabel('Frame count')
	axes[1].grid(alpha=0.25)
	axes[1].legend()

	stats_text = (
		f'Frames: {errors.size}\n'
		f'Min: {errors.min():.6f}\n'
		f'Max: {errors.max():.6f}\n'
		f'Mean: {errors.mean():.6f}\n'
		f'Std: {errors.std():.6f}'
	)
	fig.text(
		0.82,
		0.57,
		stats_text,
		fontsize=10,
		bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': '#cccccc'},
	)
	return fig


def main() -> None:
	args = parse_args()
	errors = load_error_array(args.input)
	figure = build_figure(errors, args.input)

	if args.save is not None:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(args.save, dpi=160)
		print(f'Saved visualization to {args.save}')

	if not args.no_show:
		plt.show()


if __name__ == '__main__':
	main()
