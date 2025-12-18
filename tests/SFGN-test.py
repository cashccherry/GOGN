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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-true" + ".png"))
torch.save(m_true, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-true.pt'))

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

parameters     = optParams(maxiter = 500, maxsim = 100, maxstep=.05, targetx = m_true, ltol = 1e-10, gtol=1e-10)

D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu *  m) * simPars.dx

#D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu * prec * m) * simPars.dx


if os.path.isfile("D2lo.pt"):
  D2lo = torch.load("D2lo.pt")
else:
  D2   = torch.stack([D(D(1.0*(torch.arange(simPars.nx**2) == i))) for i in range(simPars.nx**2)])
  D2lo = torch.linalg.cholesky(D2)
  torch.save(D2lo, "D2lo.pt")

alpha = torch.zeros(simPars.nx, simPars.nx)
beta  = torch.zeros(simPars.nx, simPars.nx)

alpha[1:-1,1:-1] = 20
beta[1:-1,1:-1] = 8

alpha[1:-1,0] = 12
beta[1:-1, 0] = 6
alpha[1:-1,-1] = 12
beta[1:-1, -1] = 6
alpha[0, 1:-1] = 12
beta[0, 1:-1] = 6
alpha[-1,1:-1] = 12
beta[-1,1:-1] = 6

alpha[0,0] = 6
beta[0,0] = 4
alpha[0,-1] = 6
beta[0,-1] = 4
alpha[-1,0] = 6
beta[-1,0] = 4
alpha[-1,-1] = 6
beta[-1,-1] = 4

D2diag = sigma**-2 * torch.flatten(alpha * simPars.dx**-2 + beta * nu + nu**2 * simPars.dx**2)

#plt.matshow(alpha)
#plt.colorbar()
#plt.show()
#plt.matshow(beta)
#plt.colorbar()
#
#plt.show()
#plt.matshow(D2diag.reshape(alpha.shape))
#plt.colorbar()
#
#plt.show()


print("Running CG")
f       = waveformMisfit(F, d_obs, weights)
mCG, historyCG, = CG(f, m0, parameters, D=D, h = prec, print_debug=True, line_search=line_search_interpolate_grad)

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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-{sigma_noise}-CG.png"))
historyCG.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-CG.csv')) 
torch.save(mCG, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-CG.pt'))
 
print("Running LBFGS")
f       = waveformMisfit(F, d_obs, weights)
mLBFGS, historyLBFGS, bfgsHist, _ = LBFGS(f, m0, parameters, D=D, h = prec, bfgsMemory=10, print_debug=True, line_search=line_search_interpolate_grad)

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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-{sigma_noise}-LBFGS.png"))
historyLBFGS.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-LBFGS.csv'))
torch.save(mLBFGS, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-LBFGS.pt'))
 

print("Running SFGN")
fs               = [waveformMisfit(Fs[i], d_obs[[i]], weights[[i]]) for i in range(len(Fs))]
mSFGN, historySFGN = SFGN(fs, m0, parameters, D=D, D2lo=D2lo, h=prec, print_debug=True, cgtol=1e-3, line_search=line_search_interpolate_grad)

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
fig.suptitle(f"SFGN, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-{sigma_noise}-SFGN.png"))
historySFGN.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-SFGN.csv'))
torch.save(mSFGN, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-SFGN.pt'))

print("Running GNCG")
residual = fullResidual(F, d_obs, weights)
mGNCG, historyGNCG = GNCG(residual, m0, parameters, eta0=.8, D=D, h=prec, print_debug=True, maxcgiter=20, line_search=line_search_interpolate_grad)

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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-{sigma_noise}-GNCG.png"))
historyGNCG.writeCSV(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-GNCG.csv'))
torch.save(mGNCG, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-GNCG.pt'))
 

