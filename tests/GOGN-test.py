"""Run four optimizers for acoustic experiments; see README for arguments."""
import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUTPUT_ROOT = Path(os.environ.get("GOGN_OUTPUT_DIR", REPO_ROOT)).expanduser().resolve()

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

def savefig_overwrite(figure, filename, *args, **kwargs):
    filename = Path(filename)
    if filename.exists():
        filename.chmod(0o666)
        filename.unlink()
    figure.savefig(filename, *args, **kwargs)

from gogn.objectives.FWI import *
from gogn.optimization.utils import line_search_interpolate_grad, line_search_interpolate, line_search_backtrack
from gogn.optimization.nonlinear import *
from gogn.optimization.utils import conjugate_gradient 

from gogn.regularization.quadratic import QuadraticRegularizer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("coverage", choices=["good", "bad", "good25", "bad25"])
parser.add_argument("noise", type=float, help="noise level used by the experiment")
args = parser.parse_args()
coverage = args.coverage
sigma_noise = args.noise

for directory in (OUTPUT_ROOT / "results" / "acoustic", OUTPUT_ROOT / "figures" / "acoustic"):
    directory.mkdir(parents=True, exist_ok=True)


src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

simPars      = SimulationPars(src = src, rec = rec)
m_true       = simPars.load_image_as_velocity_field()
F, d_obs, Fs = simPars.get_F_data(m_true, sigma=sigma_noise)
weights      = simPars.get_geom_weights(d_obs, invPars.kernel)
weights /= torch.max(weights)


sigma = 1.7e-5 #/ sigma_noise
nu    = 1e-8
tolerance = .005

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(m_true.detach(), plt.gca(), fontsize = 16)
simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"True Model, {coverage.capitalize()} Coverage", fontsize=18)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', f"{coverage}-true" + ".png"))
torch.save(m_true, os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-true.pt'))

m0        = 0*m_true.clone()
m0.requires_grad_()

f       = waveformMisfit(F, d_obs, weights)

loss    = f(m0)
grad    = torch.autograd.grad(loss, m0, create_graph=True)[0]
gradsum = torch.sum(grad)
prec    = torch.autograd.grad(gradsum, m0)[0].flatten()
prec   -= torch.min(prec)
prec   += 1e-1


parameters     = optParams(maxiter = 500, maxsim = 100, maxstep=.05, targetx = m_true, ltol = 1e-10, gtol=1e-10)

D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu *  m) * simPars.dx


# Cache depends on the grid, regularizer coefficients, and arithmetic precision.
cache_dir = OUTPUT_ROOT / "results" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
cache_file = cache_dir / f"D2lo-{simPars.nx}-{simPars.dx}-{sigma}-{nu}-{m0.dtype}.pt"
if cache_file.is_file():
  D2lo = torch.load(cache_file)
else:
  D2   = torch.stack([D(D(1.0*(torch.arange(simPars.nx**2) == i))) for i in range(simPars.nx**2)])
  D2lo = torch.linalg.cholesky(D2)
  torch.save(D2lo, cache_file)

reg = QuadraticRegularizer(D = D, m0 = m0, D2lo = D2lo)



print("Running CG")
f       = waveformMisfit(F, d_obs, weights)
mCG, historyCG, = CG(f, m0, parameters, reg=reg, h = prec, print_debug=True, line_search = line_search_interpolate_grad)

print(historyCG.report)

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(mCG.detach(), plt.gca(), fontsize = 16)
simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"CG, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', f"{coverage}-{sigma_noise}-CG.png"))
historyCG.writeCSV(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-CG.csv')) 
torch.save(mCG, os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-CG.pt'))
 
print("Running LBFGS")
f       = waveformMisfit(F, d_obs, weights)
mLBFGS, historyLBFGS, bfgsHist, _ = LBFGS(f, m0, parameters, reg = reg, h = prec, bfgsMemory=10, print_debug=True, line_search = line_search_interpolate_grad)

print(historyLBFGS.report)

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(mLBFGS.detach(), plt.gca(), fontsize = 16)
simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"LBFGS, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', f"{coverage}-{sigma_noise}-LBFGS.png"))
historyLBFGS.writeCSV(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-LBFGS.csv'))
torch.save(mLBFGS, os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-LBFGS.pt'))
 

print("Running GOGN")
fs               = [waveformMisfit(Fs[i], d_obs[[i]], weights[[i]]) for i in range(len(Fs))]
mSFGN, historySFGN = GOGN(fs, m0, parameters, reg=reg, h=prec, print_debug=True, cgtol=1e-3, line_search=line_search_interpolate_grad)

print(historySFGN.report)

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(mSFGN.detach(), plt.gca(), fontsize = 16)
simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"GOGN, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', f"{coverage}-{sigma_noise}-SFGN.png"))
historySFGN.writeCSV(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-SFGN.csv'))
torch.save(mSFGN, os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-SFGN.pt'))



print("Running GNCG")
residual = fullResidual(F, d_obs, weights)
mGNCG, historyGNCG = GNCG(residual, m0, parameters, eta0=.8, reg=reg, h=prec, print_debug=True, maxcgiter=20, line_search=line_search_interpolate_grad)

print(historyGNCG.report)

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(mGNCG.detach(), plt.gca(), fontsize = 16)
simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"GNCG, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', f"{coverage}-{sigma_noise}-GNCG.png"))
historyGNCG.writeCSV(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-GNCG.csv'))
torch.save(mGNCG, os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-GNCG.pt'))
 
