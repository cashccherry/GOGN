"""Small CPU acoustic inversion with all four paper optimizers.

This deliberately reduced problem checks execution, finite reconstructions, and
objective reduction. It does not reproduce a paper figure or certify convergence.
Run: python tests/smoke_inversion.py [--output PATH]
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gogn.objectives.FWI import SimulationPars, waveformMisfit, fullResidual
from gogn.optimization.nonlinear import CG, LBFGS, GOGN, GNCG, optParams
from gogn.optimization.utils import line_search_interpolate_grad
from gogn.regularization.quadratic import QuadraticRegularizer
from gogn.regularization.totalvariation import TVRegularizer


def overwrite_existing(path):
    path = Path(path)
    if path.exists():
        path.chmod(0o666)
        path.unlink()
    return path


def run_smoke(output=None, regularizer='identity'):
    torch.manual_seed(3906)
    torch.set_num_threads(1)
    src = torch.tensor([[96000, 96000], [360000, 96000]])
    rec = torch.tensor([[96000, 192000], [192000, 264000], [360000, 288000]])
    sim = SimulationPars(src=src, rec=rec, nx=20, nt=100, dt=10, freq=0.01)
    target = sim.load_image_as_velocity_field()
    F, data, Fs = sim.get_F_data(target, sigma=0.0)
    weights = torch.linalg.vector_norm(data, dim=2, keepdim=True).reciprocal()
    initial = torch.zeros_like(target, requires_grad=True)
    # Optional small smoothing operator exercises a non-diagonal Cholesky factor.
    # This is a smoke-test scale, not the full paper's regularization strength.
    if regularizer == 'identity':
        D = lambda x: 0.1*x
        factor = torch.eye(initial.numel())*0.1
    elif regularizer == 'smoothing':
        D = lambda x: 0.1*x - 0.025*sim.dx**2*sim.laplacian(x)
        basis = torch.eye(initial.numel())
        hessian = torch.stack([D(D(v)) for v in basis], dim=1)
        factor = torch.linalg.cholesky(hessian)
    elif regularizer != 'tv':
        raise ValueError(f'Unknown smoke regularizer: {regularizer}')
    if regularizer == 'tv':
        reg = TVRegularizer(m0=initial.detach(), Dx=sim.Dx, Dy=sim.Dy,
                            Dxt=sim.Dxt, Dyt=sim.Dyt, L=sim.laplacian,
                            ε=1e-7, λ=0.4, μ=1e-4, ν=0., h=sim.dx)
    else:
        reg = QuadraticRegularizer(m0=initial.detach(), D=D, D2lo=factor)
    params = optParams(maxiter=3, maxsim=30, maxstep=0.02, targetx=target)
    report = {}
    models, histories = {}, {}
    for name in ('CG', 'LBFGS', 'SFGN', 'GNCG'):
        start = time.perf_counter()
        kwargs = dict(reg=reg, h=torch.ones_like(initial), print_debug=False,
                      line_search=line_search_interpolate_grad)
        if name == 'SFGN':
            misfits = [waveformMisfit(Fi, data[[i]], weights[[i]]) for i, Fi in enumerate(Fs)]
            model, history = GOGN(misfits, initial, params, **kwargs)
        elif name == 'GNCG':
            model, history = GNCG(fullResidual(F, data, weights), initial, params,
                                  maxcgiter=3, eta0=0.8, **kwargs)
        else:
            objective = waveformMisfit(F, data, weights)
            result = (CG if name == 'CG' else LBFGS)(objective, initial, params, **kwargs)
            model, history = result[:2]
        with torch.no_grad():
            objective = waveformMisfit(F, data, weights)
            before = float(objective(initial.detach()) + reg(initial.detach()))
            after = float(objective(model.detach()) + reg(model.detach()))
        assert torch.isfinite(model).all(), name
        assert 0 <= after < before, (name, before, after)
        assert len(history.loss) >= 2, name
        assert history.forwards[-1] > 0 and history.backwards[-1] > 0
        report[name] = dict(initial_objective=before, final_objective=after,
                            forwards=history.forwards[-1], backwards=history.backwards[-1],
                            seconds=time.perf_counter()-start)
        models[name] = model.detach()
        histories[name] = history
        if output:
            dest = Path(output)/'results'/'acoustic'
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(model.detach(), dest/f'smoke-0.0-{name}.pt')
            history.writeCSV(dest/f'smoke-0.0-{name}.csv')
    if output:
        torch.save(target, Path(output)/'results'/'acoustic'/'smoke-true.pt')
        (Path(output)/'smoke.json').write_text(json.dumps(report, indent=2)+'\n')
        plot_smoke(sim, target, models, histories, Path(output)/'figures'/'smoke')
    return report


def plot_smoke(sim, target, models, histories, destination):
    """Plot the tiny inversion without changing its numerical settings."""
    import matplotlib.pyplot as plt
    import numpy as np

    destination.mkdir(parents=True, exist_ok=True)
    labels = {'CG': 'NLCG', 'LBFGS': 'L-BFGS', 'SFGN': 'GOGN', 'GNCG': 'GN-CG'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout='constrained')
    panels = [('Target', target), ('Initial model', torch.zeros_like(target))]
    panels += [(labels[name]+' · 2 updates', model) for name, model in models.items()]
    limit = 100*max(float(model.abs().max()) for _, model in panels)
    for ax, (title, model) in zip(axes.flat, panels):
        im = ax.pcolormesh(sim.X.numpy()/1000, sim.Y.numpy()/1000,
                          100*model.reshape(sim.nx, sim.nx).cpu().numpy(),
                          cmap='RdBu', vmin=-limit, vmax=limit, shading='auto')
        ax.set(title=title, xlabel='x [km]', ylabel='z [km]', aspect='equal')
        if title == 'Target':
            sim.plot_src_rec(ax)
            ax.legend(fontsize=8, loc='upper right')
    fig.colorbar(im, ax=axes, label='Speed perturbation [%]', shrink=0.85)
    fig.suptitle('Smoke inversion: target, initial model, and reconstructions', fontsize=16)
    fig.savefig(overwrite_existing(destination/'reconstructions.png'), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    for name, history in histories.items():
        updates = np.arange(len(history.loss))
        objective = np.asarray(history.loss)/history.loss[0]
        error = np.asarray(history.targetdist)/history.targetdist[0]
        work = np.asarray(history.forwards)+np.asarray(history.backwards)
        for ax, x, y in [(axes[0], updates, objective), (axes[1], updates, error),
                          (axes[2], work, objective)]:
            ax.plot(x, y, marker='o', label=labels[name])
    axes[0].set(xlabel='Optimization updates', ylabel='Relative objective', yscale='log')
    axes[1].set(xlabel='Optimization updates', ylabel='Relative model error')
    axes[2].set(xlabel='Forward + backward batch passes', ylabel='Relative objective', yscale='log')
    for ax in axes[:2]:
        ax.set_xticks([0, 1, 2])
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle('Two-update smoke check: fitting the data does not imply recovering the model', fontsize=14)
    fig.savefig(overwrite_existing(destination/'convergence.png'), dpi=180)
    plt.close(fig)
    print(f'Smoke plots saved to {destination.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='save tensors, histories, JSON, and smoke plots under this root')
    parser.add_argument('--regularizer', choices=['identity', 'smoothing', 'tv'], default='identity')
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output, args.regularizer), indent=2))
