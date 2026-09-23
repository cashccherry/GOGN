"""Plot saved acoustic experiments; see README for arguments."""
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("coverage", choices=["good", "bad", "good25", "bad25"])
parser.add_argument("noise", type=float, help="noise level used by the experiment")
args = parser.parse_args()
coverage = args.coverage
sigma_noise = args.noise

for directory in (OUTPUT_ROOT / "results" / "acoustic", OUTPUT_ROOT / "figures" / "acoustic"):
    directory.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "figures" / "acoustic" / coverage / sys.argv[2]).mkdir(parents=True, exist_ok=True)


with open(REPO_ROOT / 'tests' / 'DPI', 'r') as file:
  DPI = int(file.readline())



def alglabel(alg):
  if alg == "CG":
    return "NLCG"
  if alg == "SFGN":
    return "GOGN"
  if alg == "GNCG":
    return "GN-CG"
  if alg == "LBFGS":
    return "L-BFGS"
  return alg

def covglabel(covg):
  if covg[:4] == "good":
    return "Uniform"
  if covg[:3] == "bad":
    return "Realistic"
  return covg 

print(covglabel(coverage))

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

simPars      = SimulationPars(src = src, rec = rec)
m_true       = torch.load(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-true.pt'))

mCG = torch.load(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-CG.pt'))

mLBFGS = torch.load(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-LBFGS.pt'))

mSFGN = torch.load(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-SFGN.pt'))

mGNCG = torch.load(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-GNCG.pt'))

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(m_true.detach(), plt.gca(), fontsize = 16)
simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=18)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"{covglabel(coverage)} Coverage", fontsize=24)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', f"{coverage}-true" + ".png"), dpi=DPI)

plt.close('all')

fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

simPars.plot_model(mCG.detach(),   axes[0,0], fontsize = 16)
axes[0,0].set_title("NLCG",    fontsize=24)
simPars.plot_model(mLBFGS.detach(),   axes[0,1], fontsize = 16)
axes[0,1].set_title("L-BFGS", fontsize=24)
simPars.plot_model(mSFGN.detach(),   axes[1,0], fontsize = 16)
axes[1,0].set_title("GOGN",  fontsize=24)
simPars.plot_model(mGNCG.detach(),   axes[1,1], fontsize = 16)
axes[1,1].set_title("GN-CG", fontsize=24)

for ax in axes.flatten():
  ax.set_aspect('equal')

fig.suptitle(f"{covglabel(coverage)} Coverage, σ_noise = {sigma_noise}", fontsize=32)
savefig_overwrite(plt.gcf(), os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "reconstructions" + ".png"), dpi = DPI)

import csv

results = {}

for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  # Initialize empty lists
  iterations = []
  times = []
  forwards = []
  backwards = []
  sims = []
  objectives = []
  gradnorms = []
  model_errors = []
  steplengths = []
  
  with open(os.path.join(OUTPUT_ROOT,'results', 'acoustic', f'{coverage}-{sigma_noise}-{alg}.csv'), 'r') as csvfile:
      reader = csv.DictReader(csvfile)
      
      for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
          try:
              iterations.append(int(row['Iteration']))
              times.append(float(row['Time']))
              forwards.append(int(row['Forwards']))
              backwards.append(int(row['Backwards']))
              sims.append(int(row['Forwards']) + int(row['Backwards']))
              objectives.append(float(row['Objective']))
              gradnorms.append(float(row['Gradnorm']))
              model_errors.append(float(row['Model Error']))
              steplengths.append(float(row['Steplength']))
          except (ValueError, KeyError) as e:
              print(f"Warning: Issue with row {row_num}: {e}")
              print(f"Row data: {row}")
  
  print(f"\nSuccessfully read {len(iterations)} rows")

  results[f"iterations-{alg}"] = iterations
  results[f"times-{alg}"] = times
  results[f"forwards-{alg}"] = forwards
  results[f"backwards-{alg}"] = backwards
  results[f"sims-{alg}"] = sims 
  results[f"objectives-{alg}"] = [o/objectives[0] for o in objectives]
  results[f"gradnorms-{alg}"] = [g/gradnorms[0] for g in gradnorms]
  results[f"model_errors-{alg}"] = [m/model_errors[0] for m in model_errors]
  results[f"steplengths-{alg}"] = steplengths




title_fontsize = 20
fontsize = 20
labelsize=15
shade = 0.3
std_constant = 3.0

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"][:-1], results[f"steplengths-{alg}"][:-1], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Steplength", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "iter-step" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"], results[f"objectives-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Objective Function", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "iter-obj" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"sims-{alg}"][-1])
  ax.loglog(results[f"sims-{alg}"], results[f"objectives-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("PDE Solves", fontsize=labelsize)
plt.ylabel("Objective Function", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "work-obj" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"times-{alg}"][-1])
  ax.loglog(results[f"times-{alg}"], results[f"objectives-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Times", fontsize=labelsize)
plt.ylabel("Objective Function", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "time-obj" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"], results[f"model_errors-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Model Error", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "iter-err" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"sims-{alg}"][-1])
  ax.loglog(results[f"sims-{alg}"], results[f"model_errors-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("PDE Solves", fontsize=labelsize)
plt.ylabel("Model Error", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "work-err" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"times-{alg}"][-1])
  ax.loglog(results[f"times-{alg}"], results[f"model_errors-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Times", fontsize=labelsize)
plt.ylabel("Model Error", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "time-err" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"], results[f"gradnorms-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Gradient Norm", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "iter-grad" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"sims-{alg}"][-1])
  ax.loglog(results[f"sims-{alg}"], results[f"gradnorms-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("PDE Solves", fontsize=labelsize)
plt.ylabel("Gradient Norm", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "work-grad" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"times-{alg}"][-1])
  ax.loglog(results[f"times-{alg}"], results[f"gradnorms-{alg}"], linewidth=2, marker='o', markersize=8, label=alglabel(alg));
plt.legend()
plt.xlabel("Times", fontsize=labelsize)
plt.ylabel("Gradient Norm", fontsize=labelsize)
savefig_overwrite(fig1, os.path.join(OUTPUT_ROOT, 'figures',  'acoustic', coverage, sys.argv[2], "time-grad" + ".png"), dpi=DPI , bbox_inches="tight", pad_inches=0.0)

