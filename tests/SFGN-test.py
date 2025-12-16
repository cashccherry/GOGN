import os
import sys

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

from sfgn.objectives.FWI import *
from sfgn.optimization.utils import line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *

coverage = sys.argv[1]

maxiter  = 15

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

simPars      = SimulationPars(src = src, rec = rec)
m_true       = simPars.load_image_as_velocity_field()
F, d_obs, Fs = simPars.get_F_data(m_true)
weights      = simPars.get_geom_weights(d_obs, invPars.kernel)
weights /= torch.max(weights)


sigma = 1.7e-5
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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-true" + ".png"))

m0        = 0*m_true.clone()
m0.requires_grad_()

f       = waveformMisfit(F, d_obs, weights)

loss    = f(m0)
grad    = torch.autograd.grad(loss, m0, create_graph=True)[0]
gradsum = torch.sum(grad)
prec    = torch.autograd.grad(gradsum, m0)[0].flatten()
prec   -= torch.min(prec)
prec   += 1e-1

#print(torch.min(prec),torch.max(prec))
#quit()

parameters     = optParams(maxiter = maxiter, maxstep=.05, targetx = m_true, ltol = 1e-10)

D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu *  m) * simPars.dx

print("Running CG")
f       = waveformMisfit(F, d_obs, weights)
mCG, historyCG, = CG(f, m0, parameters, D=D, h = prec, print_debug=True)

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
fig.suptitle(f"CG Recovered Model, {coverage.capitalize()} Coverage", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-CG.png"))
historyCG.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-CG.csv')) 
torch.save(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-CG.pt'), mCG)
 
print("Running LBFGS")
f       = waveformMisfit(F, d_obs, weights)
mLBFGS, historyLBFGS, bfgsHist, _ = LBFGS(f, m0, parameters, D=D, h = prec, bfgsMemory=50, print_debug=True)

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
fig.suptitle(f"LBFGS Recovered Model, {coverage.capitalize()} Coverage", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-LBFGS.png"))
historyLBFGS.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-LBFGS.csv'))
torch.save(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-LBFGS.pt'))
 

print("Running SFGN")
fs               = [waveformMisfit(Fs[i], d_obs[[i]], weights[[i]]) for i in range(len(Fs))]
mSFGN, historySFGN = SFGN(fs, m0, parameters, D=D, h=prec, print_debug=True)

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
fig.suptitle(f"SFGN Recovered Model, {coverage.capitalize()} Coverage", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-SFGN.png"))
historySFGN.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-SFGN.csv'))
torch.save(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-SFGN.pt'))
 

print("Running GNCG")
residual = fullResidual(F, d_obs, weights)
mGNCG, historyGNCG = GNCG(residual, m0, parameters, eta0=.8, D=D, h=prec, print_debug=True, maxcgiter=20)

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
fig.suptitle(f"GNCG Recovered Model, {coverage.capitalize()} Coverage", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-SFGN.png"))
historySFGN.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-SFGN.csv'))
torch.save(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-SFGN.pt'))
 

