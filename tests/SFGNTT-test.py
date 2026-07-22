import os
import sys

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

from sfgn.objectives.FWITT import *
from sfgn.optimization.utils import line_search_interpolate_grad, line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *
from sfgn.optimization.utils import conjugate_gradient 


coverage    = sys.argv[1]
sigma_noise = float(sys.argv[2])
norm_p      = float(sys.argv[3])


src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

simPars      = SimulationPars(src = src, rec = rec)
m_true       = simPars.load_image_as_velocity_field()
F, Fs = simPars.get_F_data(m_true)
d_obs = torch.randn_like(F(m_true))*sigma_noise
weights      = simPars.get_geom_weights(d_obs, invPars.kernel)
weights /= torch.max(weights)

#weights /= sigma_noise 
#print(torch.histogram(weights))

sigma = 9e-6 #/ sigma_noise
nu    = 7e-9
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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acousticTT', f"{coverage}-true" + ".png"))
torch.save(m_true, os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-true.pt'))

m0        = 0*m_true.clone()
m0.requires_grad_()

f       = waveformMisfit(F, d_obs, lambda m : weights * m, p = norm_p)

loss    = f(m0)
grad    = torch.autograd.grad(loss, m0, create_graph=True)[0]
gradsum = torch.sum(grad)
prec    = torch.autograd.grad(gradsum, m0)[0].flatten()
prec   -= torch.min(prec)

#q = .01
#prec[prec < torch.quantile(prec, q)] = torch.quantile(prec, q)
prec   += 1e1

print(torch.min(prec),torch.max(prec))
#quit()

parameters     = optParams(maxiter = 100, maxsim = 100, maxstep=.03, targetx = m_true, ltol = 1e-10, gtol=1e-10)

D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu *  m) * simPars.dx

#D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu * prec * m) * simPars.dx

fig, axes = plt.subplots(1,3,figsize=(13, 4), constrained_layout = True)
im = simPars.plot_std(prec.detach(), axes[0])
simPars.plot_src_rec(axes[0])
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].set_title('Preconditioner')

n = torch.randn_like(prec)
n = torch.outer(*2*[2.0*((torch.arange(simPars.nx) // 5) % 2) - 1]).flatten()
n = grad.detach()

im = simPars.plot_grad(n.detach(), axes[1])
axes[1].set_aspect('equal')
axes[1].set_title('Gradient')
cbar = plt.colorbar(im)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()

im = simPars.plot_grad(conjugate_gradient(lambda m : D(D(m)) + prec * m, n, tol=.005, max_iter=10000)[0].detach(), axes[2])
axes[2].set_aspect('equal')
axes[2].set_title('Preconditioned Gradient')
cbar = plt.colorbar(im)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()

fig.suptitle(f"Effect of Preconditioning", fontsize=18)
#plt.show()

plt.savefig(os.path.join(os.getcwd(), 'figures',  'acousticTT', f"{coverage}-{norm_p}-prec" + ".png"))
torch.save(m_true, os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{norm_p}-true.pt'))

plt.close('all')
#quit()

if os.path.isfile("D2loTT.pt"):
  D2lo = torch.load("D2loTT.pt")
else:
  D2   = torch.stack([D(D(1.0*(torch.arange(simPars.nx**2) == i))) for i in range(simPars.nx**2)])
  D2lo = torch.linalg.cholesky(D2)
  torch.save(D2lo, "D2loTT.pt")

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
f       = waveformMisfit(F, d_obs, lambda m : weights*m, p = norm_p)
mCG, historyCG, = CG(f, m0, parameters, D=D, h = prec, print_debug=True, line_search = line_search_interpolate_grad)

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
fig.suptitle(f"CG, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}, p = {norm_p}", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acousticTT', f"{coverage}-{sigma_noise}-{norm_p}-CG.png"))
historyCG.writeCSV(os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-CG.csv')) 
torch.save(mCG, os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-CG.pt'))
 
print("Running LBFGS")
f       = waveformMisfit(F, d_obs, lambda m : weights*m, p = norm_p)
mLBFGS, historyLBFGS, bfgsHist, _ = LBFGS(f, m0, parameters, D=D, h = prec, bfgsMemory=10, print_debug=True, line_search = line_search_interpolate_grad)

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
fig.suptitle(f"LBFGS, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}, p = {norm_p}", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acousticTT', f"{coverage}-{sigma_noise}-{norm_p}-LBFGS.png"))
historyLBFGS.writeCSV(os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-LBFGS.csv'))
torch.save(mLBFGS, os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-LBFGS.pt'))
 

print("Running SFGN")
fs               = [waveformMisfit(Fs[i], d_obs[[i]], lambda m : weights[[i]]*m, p = norm_p) for i in range(len(Fs))]
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
fig.suptitle(f"SFGN, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}, p = {norm_p}", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acousticTT', f"{coverage}-{sigma_noise}-{norm_p}-SFGN.png"))
historySFGN.writeCSV(os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-SFGN.csv'))
torch.save(mSFGN, os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-SFGN.pt'))

print("Running GNCG")
residual = fullResidual(F, d_obs, lambda m : weights*m, p = norm_p)
mGNCG, historyGNCG = GNCG(residual, m0, parameters, eta0=.8, D=D, D2diag=D2diag, h=prec, print_debug=True, maxcgiter=20, line_search=line_search_interpolate_grad)

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
fig.suptitle(f"GNCG, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}, p = {norm_p}", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'acousticTT', f"{coverage}-{sigma_noise}-{norm_p}-GNCG.png"))
historyGNCG.writeCSV(os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-GNCG.csv'))
torch.save(mGNCG, os.path.join(os.getcwd(),'results', 'acousticTT', f'{coverage}-{sigma_noise}-{norm_p}-GNCG.pt'))
 

