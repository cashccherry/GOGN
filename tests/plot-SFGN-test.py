import os
import sys

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

from sfgn.objectives.FWI import *
from sfgn.optimization.utils import line_search_interpolate_grad, line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *

coverage = sys.argv[1]
sigma_noise = float(sys.argv[2])


src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

simPars      = SimulationPars(src = src, rec = rec)
m_true       = torch.load(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-true.pt'))

mCG = torch.load(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-CG.pt'))

mLBFGS = torch.load(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-LBFGS.pt'))

mSFGN = torch.load(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-SFGN.pt'))

mGNCG = torch.load(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-GNCG.pt'))


fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

simPars.plot_model(mCG.detach(),   axes[0,0], fontsize = 16)
axes[0,0].set_title("CG",    fontsize=24)
simPars.plot_model(mLBFGS.detach(),   axes[0,1], fontsize = 16)
axes[0,1].set_title("LBFGS", fontsize=24)
simPars.plot_model(mSFGN.detach(),   axes[1,0], fontsize = 16)
axes[1,0].set_title("SFGN",  fontsize=24)
simPars.plot_model(mGNCG.detach(),   axes[1,1], fontsize = 16)
axes[1,1].set_title("GN-CG", fontsize=24)

for ax in axes.flatten():
  ax.set_aspect('equal')

fig.suptitle(f"{coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=32)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "reconstructions" + ".png"))

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
  
  with open(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-{alg}.csv'), 'r') as csvfile:
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




title_fontsize = 20
fontsize = 20
labelsize=15
shade = 0.3
std_constant = 3.0

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"], results[f"objectives-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Objective Function", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "iter-obj" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"sims-{alg}"][-1])
  ax.loglog(results[f"sims-{alg}"], results[f"objectives-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
#plt.xlim(0, np.median(np.array(xmax)))
plt.legend()
plt.xlabel("PDE Solves", fontsize=labelsize)
plt.ylabel("Objective Function", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "work-obj" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"times-{alg}"][-1])
  ax.loglog(results[f"times-{alg}"], results[f"objectives-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
#plt.xlim(0, np.median(np.array(xmax)))
plt.legend()
plt.xlabel("Times", fontsize=labelsize)
plt.ylabel("Objective Function", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "time-obj" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"], results[f"model_errors-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Model Error", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "iter-err" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"sims-{alg}"][-1])
  ax.loglog(results[f"sims-{alg}"], results[f"model_errors-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
#plt.xlim(0, np.median(np.array(xmax)))
plt.legend()
plt.xlabel("PDE Solves", fontsize=labelsize)
plt.ylabel("Model Error", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "work-err" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"times-{alg}"][-1])
  ax.loglog(results[f"times-{alg}"], results[f"model_errors-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
#plt.xlim(0, np.median(np.array(xmax)))
plt.legend()
plt.xlabel("Times", fontsize=labelsize)
plt.ylabel("Model Error", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "time-err" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  ax.semilogy(results[f"iterations-{alg}"], results[f"gradnorms-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
plt.legend()
plt.xlabel("Iterations", fontsize=labelsize)
plt.ylabel("Gradient Norm", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "iter-grad" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"sims-{alg}"][-1])
  ax.loglog(results[f"sims-{alg}"], results[f"gradnorms-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
#plt.xlim(0, np.median(np.array(xmax)))
plt.legend()
plt.xlabel("PDE Solves", fontsize=labelsize)
plt.ylabel("Gradient Norm", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "work-grad" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)

fig1 = plt.figure()
ax = plt.axes()
xmax = []
for alg in ["CG", "LBFGS", "SFGN", "GNCG"]:
  xmax.append(results[f"times-{alg}"][-1])
  ax.loglog(results[f"times-{alg}"], results[f"gradnorms-{alg}"], linewidth=2, marker='o', markersize=8, label=alg);
#plt.xlim(0, np.median(np.array(xmax)))
plt.legend()
plt.xlabel("Times", fontsize=labelsize)
plt.ylabel("Gradient Norm", fontsize=labelsize)
fig1.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', coverage, sys.argv[2], "time-grad" + ".png"), dpi=300 , bbox_inches="tight", pad_inches=0.0)


