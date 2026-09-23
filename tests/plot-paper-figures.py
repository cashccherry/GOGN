"""Assemble numbered manuscript/rebuttal figures from existing plot panels."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from gogn.objectives.FWI import CWD_file, SimulationPars, percent_formatter
from matplotlib.ticker import FuncFormatter


def overwrite_existing(path):
    path = Path(path)
    if path.exists():
        path.chmod(0o666)
        path.unlink()
    return path


def assemble(rows, destination):
    missing = [str(p) for row in rows for p in row if not p.is_file()]
    if missing:
        raise FileNotFoundError('Run the corresponding plot script first. Missing:\n'+'\n'.join(missing))
    images = [[Image.open(p).convert('RGB') for p in row] for row in rows]
    cell_width = max(im.width for row in images for im in row)
    cell_height = max(im.height for row in images for im in row)
    gap = 24
    canvas = Image.new('RGB', (len(images[0])*(cell_width+gap)-gap,
                              len(images)*(cell_height+gap)-gap), 'white')
    for y, row in enumerate(images):
        for x, im in enumerate(row):
            # Preserve each source panel's pixels and aspect ratio.
            canvas.paste(im, (x*(cell_width+gap)+(cell_width-im.width)//2,
                              y*(cell_height+gap)+(cell_height-im.height)//2))
            im.close()
    canvas.save(overwrite_existing(destination), dpi=(400, 400))


def acquisition_figure(destination):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), layout='constrained')
    for ax, coverage in zip(axes.flat, ['good', 'bad', 'good25', 'bad25']):
        folder = Path(CWD_file)/'FWI'/coverage
        sim = SimulationPars(src=torch.tensor(np.load(folder/'srcs.npy')),
                             rec=torch.tensor(np.load(folder/'recs.npy')))
        im = sim.plot_model(sim.load_image_as_velocity_field(), ax)
        sim.plot_src_rec(ax)
        ax.set_aspect('equal')
        ax.set_title(('Uniform' if coverage.startswith('good') else 'Realistic')+
                     f' coverage, {sim.n_events} sources')
        ax.legend()
    bar = fig.colorbar(im, ax=axes, format=FuncFormatter(percent_formatter))
    bar.set_label('Speed perturbation')
    fig.savefig(overwrite_existing(destination), dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuttal', action='store_true')
    parser.add_argument('--figures', nargs='+', type=int)
    args = parser.parse_args()
    output = Path(os.environ.get('GOGN_OUTPUT_DIR', ROOT)).expanduser().resolve()/'figures'
    destination = output/('rebuttal' if args.rebuttal else 'paper')
    destination.mkdir(parents=True, exist_ok=True)
    # First figure number, family, uniform/realistic coverage pair, noise, p.
    pairs = ([(1, 'acousticTT', ['good', 'bad'], '1e-2', '0.5'),
              (3, 'acousticTV', ['good', 'bad'], '1e-2', None),
              (5, 'acousticTV', ['good', 'bad'], '1', None)] if args.rebuttal else
             [(2, 'acoustic', ['good', 'bad'], '1e-1', None),
              (4, 'acoustic', ['good25', 'bad25'], '1e-1', None),
              (6, 'acoustic', ['good', 'bad'], '1e-2', None),
              (8, 'acoustic', ['good25', 'bad25'], '1e-2', None),
              (10, 'acoustic', ['good', 'bad'], '1', None),
              (12, 'acoustic', ['good25', 'bad25'], '1', None),
              (14, 'acousticTV', ['good', 'bad'], '1e-2', None),
              (16, 'acousticTV', ['good', 'bad'], '1', None),
              (18, 'acousticTV', ['good25', 'bad25'], '1e-2', None),
              (20, 'acousticTV', ['good25', 'bad25'], '1', None)])
    count = 6 if args.rebuttal else 21
    selected = set(args.figures or range(1, count+1))
    if not selected <= set(range(1, count+1)):
        parser.error(f'figure numbers must be between 1 and {count}')
    if not args.rebuttal and 1 in selected:
        acquisition_figure(destination/'figure-01.png')
    for first, family, coverages, noise, p in pairs:
        paths = [output/family/c/noise for c in coverages]
        if p is not None:
            paths = [path/p for path in paths]
        for number, filenames in [(first, ['work-err.png', 'work-grad.png', 'work-obj.png']),
                                  (first+1, ['reconstructions.png'])]:
            if number in selected:
                assemble([[path/name for name in filenames] for path in paths],
                         destination/f'figure-{number:02}.png')
    print(f'Wrote {len(selected)} figure(s) to {destination}')


if __name__ == '__main__':
    main()
