import os
import sys
from random import randint
import torch
import numpy as np
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

from sfgn.objectives.FWITT import *
from sfgn.optimization.utils import line_search_interpolate_grad, line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *
from sfgn.optimization.utils import conjugate_gradient, bfgsHistory, lbfgs_invhess, LimitedMemoryStructuredSecantTorch


coverage = "good25" #sys.argv[1]
sigma_noise = .053 #float(sys.argv[2])

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)




simPars      = SimulationPars(src = src, rec = rec)
m_true       = simPars.load_image_as_velocity_field()
F, Fs = simPars.get_F_data(m_true)
#weights      = simPars.get_geom_weights(d_obs, invPars.kernel)
#weights /= torch.max(weights)

sigma = 4e-6 #/ sigma_noise
nu    = 7e-9
tolerance = .005

print(f"Maximum Time : {simPars.dt * simPars.nt}s")
print(f"Noise STD : {sigma_noise}s")
d_obs = torch.randn_like(F(m_true))*sigma_noise

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
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{coverage}-true" + ".png"))
torch.save(m_true, os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-true.pt'))

m0        = 0*m_true.clone()
m0.requires_grad_()

f       = waveformMisfit(F, d_obs, lambda x: x/sigma_noise, p = 1)

loss    = f(m0)
print(F(m0), loss)
grad    = torch.autograd.grad(loss, m0, create_graph=True)[0]
gradsum = torch.sum(grad)
prec    = torch.autograd.grad(gradsum, m0)[0].flatten()
prec   -= torch.min(prec)
prec   += 1e-1

#print(torch.min(prec),torch.max(prec))
#quit()

parameters     = optParams(maxiter = 500, maxsim = 100, maxstep=.05, targetx = m_true, ltol = 1e-10, gtol=1e-10)

D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu *  m) * simPars.dx

fig, ax = plt.subplots(3,3, figsize=(6,5), constrained_layout = True)
ims = []
for i in range(3):
  for j in range(3):
    n = torch.randn_like(m_true)
    im = simPars.plot_grad(conjugate_gradient(lambda m : D(m), n, tol=.005, max_iter=10000)[0].detach(), ax[i,j])
    ims.append(im)
for im in ims:
  im.set_clim(vmin=-0.05, vmax=0.05)

fig.colorbar(ims[0], 
  ax=ax,                # span the entire 3x3 grid
  location='right',     # place on the right side
  shrink=0.9,           # slightly shorter than full height to avoid crowding
  pad=0.04,             # spacing from the plots
  aspect=30
)            # controls thinness (higher = thinner)
fig.suptitle("Samples from the Prior")
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{coverage}-prior" + ".png"))
plt.close()

# # #D = lambda m : sigma**-1 * (- simPars.laplacian(m) + nu * prec * m) * simPars.dx
# # 
# # fig, axes = plt.subplots(1,3,figsize=(5.25, 4), constrained_layout = True)
# # im = simPars.plot_std(prec.detach(), axes[0])
# # simPars.plot_src_rec(axes[0])
# # cbar = plt.colorbar(im)
# # cbar.set_label("% Speed Perturbation", fontsize=16)
# # cbar.formatter = FuncFormatter(percent_formatter)
# # cbar.update_ticks()
# # axes[0].legend()
# # axes[0].set_aspect('equal')
# # axes[0].set_title('Preconditioner')
# # 
# # #n = torch.randn_like(prec)
# # #n = torch.outer(*2*[2.0*((torch.arange(simPars.nx) // 5) % 2) - 1]).flatten()
# # n = grad.detach()
# # 
# # im = simPars.plot_grad(n.detach(), axes[1])
# # axes[1].set_aspect('equal')
# # axes[1].set_title('Gradient')
# # cbar = plt.colorbar(im)
# # cbar.formatter = FuncFormatter(percent_formatter)
# # cbar.update_ticks()
# # 
# # im = simPars.plot_grad(conjugate_gradient(lambda m : D(D(m)) + prec * m, n, tol=.005, max_iter=10000)[0].detach(), axes[2])
# # axes[2].set_aspect('equal')
# # axes[2].set_title('Preconditioned Gradient')
# # cbar = plt.colorbar(im)
# # cbar.formatter = FuncFormatter(percent_formatter)
# # cbar.update_ticks()
# # 
# # fig.suptitle(f"Effect of Preconditioning", fontsize=18)
# # plt.show()
# # 
# # #plt.savefig(os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-true" + ".png"))
# # #torch.save(m_true, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-true.pt'))

if os.path.isfile("D2louq.pt"):
  D2lo = torch.load("D2louq.pt")
else:
  D2   = torch.stack([D(D(1.0*(torch.arange(simPars.nx**2) == i))) for i in range(simPars.nx**2)])
  D2lo = torch.linalg.cholesky(D2)
  torch.save(D2lo, "D2louq.pt")

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


#print("Running CG")
#f       = waveformMisfit(F, d_obs, lambda x: x/sigma_noise)
#mCG, historyCG, = CG(f, m0, parameters, D=D, h = prec, print_debug=True, line_search = line_search_interpolate_grad)
#
#print(historyCG.report)
#
#fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
#im = simPars.plot_model(mCG.detach(), plt.gca(), fontsize = 16)
#simPars.plot_src_rec(plt.gca())
#cbar = plt.colorbar(im)
#cbar.set_label("% Speed Perturbation", fontsize=16)
#cbar.formatter = FuncFormatter(percent_formatter)
#cbar.update_ticks()
#plt.legend()
#plt.gca().set_aspect('equal')
#fig.suptitle(f"CG, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
#plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{coverage}-{sigma_noise}-CG.png"))
#historyCG.writeCSV(os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-CG.csv')) 
#torch.save(mCG, os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-CG.pt'))
# 
#print("Running LBFGS")
#f       = waveformMisfit(F, d_obs, lambda x: x/sigma_noise)
#mLBFGS, historyLBFGS, bfgsHist, _ = LBFGS(f, m0, parameters, D=D, h = prec, bfgsMemory=10, print_debug=True, line_search = line_search_interpolate_grad)
#
#print(historyLBFGS.report)
#
#fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
#im = simPars.plot_model(mLBFGS.detach(), plt.gca(), fontsize = 16)
#simPars.plot_src_rec(plt.gca())
#cbar = plt.colorbar(im)
#cbar.set_label("% Speed Perturbation", fontsize=16)
#cbar.formatter = FuncFormatter(percent_formatter)
#cbar.update_ticks()
#plt.legend()
#plt.gca().set_aspect('equal')
#fig.suptitle(f"LBFGS, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
#plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{coverage}-{sigma_noise}-LBFGS.png"))
#historyLBFGS.writeCSV(os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-LBFGS.csv'))
#torch.save(mLBFGS, os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-LBFGS.pt'))
# 

print("Running SFGN")
fs               = [waveformMisfit(Fs[i], d_obs[[i]], lambda x: x/sigma_noise) for i in range(len(Fs))]
mSFGN, historySFGN, Jhist = SFGNUQ(fs, m0, parameters, D=D, D2lo=D2lo, h=prec, print_debug=True, cgtol=1e-3, line_search=line_search_interpolate_grad)

print(historySFGN.report)

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
im = simPars.plot_model(mSFGN.detach(), plt.gca(), fontsize = 16)
#simPars.plot_src_rec(plt.gca())
cbar = plt.colorbar(im)
cbar.set_label("% Speed Perturbation", fontsize=16)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()
#plt.legend()
plt.gca().set_aspect('equal')
fig.suptitle(f"GOGN, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{coverage}-{sigma_noise}-GOGN.png"))
historySFGN.writeCSV(os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-GOGN.csv'))
torch.save(mSFGN, os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-GOGN.pt'))

#print("Running GNCG")
#residual = fullResidual(F, d_obs, weights)
#mGNCG, historyGNCG = GNCG(residual, m0, parameters, eta0=.8, D=D, D2diag=D2diag, h=prec, print_debug=True, maxcgiter=20, line_search=line_search_interpolate_grad)
#
#print(historyGNCG.report)
#
#fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
#im = simPars.plot_model(mGNCG.detach(), plt.gca(), fontsize = 16)
#simPars.plot_src_rec(plt.gca())
#cbar = plt.colorbar(im)
#cbar.set_label("% Speed Perturbation", fontsize=16)
#cbar.formatter = FuncFormatter(percent_formatter)
#cbar.update_ticks()
#plt.legend()
#plt.gca().set_aspect('equal')
#fig.suptitle(f"GNCG, {coverage.capitalize()} Coverage, σ_noise = {sigma_noise}", fontsize=18)
#plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{coverage}-{sigma_noise}-GNCG.png"))
#historyGNCG.writeCSV(os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-GNCG.csv'))
#torch.save(mGNCG, os.path.join(os.getcwd(),'results', 'uq', f'{coverage}-{sigma_noise}-GNCG.pt'))
 

# GOGN Approximation

dsprs = []
for i in range(10):
  index = randint(0,simPars.nx**2)
  dsprs.append(conjugate_gradient(lambda m : D(D(m)), 1.0 * (torch.arange(simPars.nx**2) == index), tol=tolerance, max_iter=10000)[0][index])

print(f"priorvar = {dsprs}")
dspr = torch.median(torch.Tensor(dsprs))

def dfpPPH(n):
  ns = conjugate_gradient(D, n, tol=.005, max_iter=10000)[0].detach()
  hns = Jhist[0][0].T @ (Jhist[0][0] @ ns)
  hnss = conjugate_gradient(D, hns, tol=.005, max_iter=10000)[0].detach()
  return hnss

sprhalf = lambda m : conjugate_gradient(D, m, tol=.005, max_iter=10000)[0]

print(f"D - {D}")

r = Jhist[0][0].shape[0]
oversamp = 5

Omega = torch.randn(simPars.nx**2, r + oversamp)
Y = torch.stack([dfpPPH(Omega[:,i]) for i in range(r + oversamp)]).T
Q, R = torch.linalg.qr(Y)
T = torch.linalg.solve((Q.T @ Omega).T, (Q.T @ Y).T)
S, Vv = eigsh(T.numpy(),k=r)
S = torch.tensor(S)
print(S)
S[S < 0] = 0
Vv = torch.tensor(Vv)
V = Q @ Vv
sprV = torch.stack([sprhalf(V[:,i]) for i in range(r)]).T

plt.figure()
im = simPars.plot_std((dspr - (sprV**2)@(S/(S+1)))**.5, plt.gca()) 
plt.colorbar(im, shrink=.6)
plt.title("GOGN Standard Deviation",fontsize=24)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-gognstd" + ".png"))
plt.close()

fig, ax = plt.subplots(3,3, figsize=(6,5), constrained_layout = True)
ims = []
for i in range(3):
  for j in range(3):
    n = torch.randn_like(m_true)
    im = simPars.plot_model((mSFGN + sprhalf(V@(((S + 1)**-.5 - 1) * (sprV.T @ n)) + n)).detach(), ax[i,j])
    ims.append(im)
#for im in ims:
#  im.set_clim(vmin=-0.2, vmax=0.2)

fig.colorbar(ims[0], 
  ax=ax,                # span the entire 3x3 grid
  location='right',     # place on the right side
  shrink=0.9,           # slightly shorter than full height to avoid crowding
  pad=0.04,             # spacing from the plots
  aspect=30
)            # controls thinness (higher = thinner)
fig.suptitle("Samples from the GOGN Posterior")
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-gognpost" + ".png"))
plt.close()


# DFP Approximation

dfpHist    = bfgsHistory()

for i in range(len(Jhist) - 1):
#  dfpHist.update(torch.sum(Jhist[i][0] - Jhist[i+1][0]), Jhist[i][2] - Jhist[i+1][2])
  dfpHist.update(Jhist[i][2] - Jhist[i+1][2], torch.sum(Jhist[i][0] - Jhist[i+1][0], dim = 0))

dsprs = []
for i in range(10):
  index = randint(0,simPars.nx**2)
  dsprs.append(conjugate_gradient(lambda m : D(D(m)), 1.0 * (torch.arange(simPars.nx**2) == index), tol=tolerance, max_iter=10000)[0][index])

print(f"priorvar = {dsprs}")
dspr = torch.median(torch.Tensor(dsprs))

def dfpPPH(n):
  ns = conjugate_gradient(D, n, tol=.005, max_iter=10000)[0].detach()
  hns = lbfgs_invhess(ns, dfpHist, init= lambda g : g * prec)
  hnss = conjugate_gradient(D, hns, tol=.005, max_iter=10000)[0].detach()
  return hnss

sprhalf = lambda m : conjugate_gradient(D, m, tol=.005, max_iter=10000)[0]

print(f"D - {D}")

r = 80
oversamp = 5

Omega = torch.randn(simPars.nx**2, r + oversamp)
Y = torch.stack([dfpPPH(Omega[:,i]) for i in range(r + oversamp)]).T
Q, R = torch.linalg.qr(Y)
T = torch.linalg.solve((Q.T @ Omega).T, (Q.T @ Y).T)
S, Vv = eigsh(T.numpy(),k=r)
S = torch.tensor(S)
print(S)
S[S < 0] = 0
Vv = torch.tensor(Vv)
V = Q @ Vv
sprV = torch.stack([sprhalf(V[:,i]) for i in range(r)]).T

plt.figure()
im = simPars.plot_std((dspr - (sprV**2)@(S/(S+1)))**.5, plt.gca()) 
plt.colorbar(im, shrink=.6)
plt.title("DFP Standard Deviation",fontsize=24)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-dfpstd" + ".png"))
plt.close()

fig, ax = plt.subplots(3,3, figsize=(6,5), constrained_layout = True)
ims = []
for i in range(3):
  for j in range(3):
    n = torch.randn_like(m_true)
    im = simPars.plot_model((mSFGN + sprhalf(V@(((S + 1)**-.5 - 1) * (sprV.T @ n)) + n)).detach(), ax[i,j])
    ims.append(im)
#for im in ims:
#  im.set_clim(vmin=-0.2, vmax=0.2)

fig.colorbar(ims[0], 
  ax=ax,                # span the entire 3x3 grid
  location='right',     # place on the right side
  shrink=0.9,           # slightly shorter than full height to avoid crowding
  pad=0.04,             # spacing from the plots
  aspect=30
)            # controls thinness (higher = thinner)
fig.suptitle("Samples from the DFP Posterior")
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-dfppost" + ".png"))
plt.close()


# Multiple DFP Approximation
N = Jhist[0][0].shape[0]
dfpHists    = [bfgsHistory() for _ in range(N)]

for i in range(len(Jhist) - 1):
#  dfpHist.update(torch.sum(Jhist[i][0] - Jhist[i+1][0]), Jhist[i][2] - Jhist[i+1][2])
  for j in range(N):
    dfpHists[j].update(Jhist[i][2] - Jhist[i+1][2], (Jhist[i][0] - Jhist[i+1][0])[j,:])

def dfpPPH(n):
  ns = conjugate_gradient(D, n, tol=.005, max_iter=10000)[0].detach()
  hns = 0
  for j in range(N):
    hns += lbfgs_invhess(ns, dfpHists[j], init= lambda g : g * prec/N)
  hnss = conjugate_gradient(D, hns, tol=.005, max_iter=10000)[0].detach()
  return hnss

sprhalf = lambda m : conjugate_gradient(D, m, tol=.005, max_iter=10000)[0]

print(f"D - {D}")

r = 80
oversamp = 5

Omega = torch.randn(simPars.nx**2, r + oversamp)
Y = torch.stack([dfpPPH(Omega[:,i]) for i in range(r + oversamp)]).T
Q, R = torch.linalg.qr(Y)
T = torch.linalg.solve((Q.T @ Omega).T, (Q.T @ Y).T)
S, Vv = eigsh(T.numpy(),k=r)
S = torch.tensor(S)
print(S)
S[S < 0] = 0
Vv = torch.tensor(Vv)
V = Q @ Vv
sprV = torch.stack([sprhalf(V[:,i]) for i in range(r)]).T

plt.figure()
im = simPars.plot_std((dspr - (sprV**2)@(S/(S+1)))**.5, plt.gca()) 
plt.colorbar(im, shrink=.6)
plt.title("mDFP Standard Deviation",fontsize=24)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-mdfpstd" + ".png"))
plt.close()

fig, ax = plt.subplots(3,3, figsize=(6,5), constrained_layout = True)
ims = []
for i in range(3):
  for j in range(3):
    n = torch.randn_like(m_true)
    im = simPars.plot_model((mSFGN + sprhalf(V@(((S + 1)**-.5 - 1) * (sprV.T @ n)) + n)).detach(), ax[i,j])
    ims.append(im)
#for im in ims:
#  im.set_clim(vmin=-0.2, vmax=0.2)

fig.colorbar(ims[0], 
  ax=ax,                # span the entire 3x3 grid
  location='right',     # place on the right side
  shrink=0.9,           # slightly shorter than full height to avoid crowding
  pad=0.04,             # spacing from the plots
  aspect=30
)            # controls thinness (higher = thinner)
fig.suptitle("Samples from the mDFP Posterior")
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-mdfppost" + ".png"))
plt.close()


# High Res GOGN Approximation
N = Jhist[0][0].shape[0]
S_approx = LimitedMemoryStructuredSecantTorch(
    n=Jhist[0][0].shape[1],
    m=12,
    gamma=0.5,
    update_gamma=True,
    dtype=torch.float64,
)

for i in range(len(Jhist) - 1):
  s = Jhist[i+1][2] - Jhist[i][2]
  y = Jhist[i+1][0].T @ Jhist[i+1][3] - Jhist[i][0].T @ Jhist[i][3]
  y_hash = Jhist[i+1][0].T @ Jhist[i+1][3] - Jhist[i][0].T @ Jhist[i+1][3]
  S_approx.update(s, y, y_hash)
#  dfpHist.update(torch.sum(Jhist[i][0] - Jhist[i+1][0]), Jhist[i][2] - Jhist[i+1][2])
#  for j in range(N):
#    dfpHists[j].update(Jhist[i][2] - Jhist[i+1][2], (Jhist[i][0] - Jhist[i+1][0])[j,:])

def dfpPPH(n):
  ns = conjugate_gradient(D, n, tol=.005, max_iter=10000)[0].detach()
  hns = Jhist[0][0].T @ (Jhist[0][0] @ ns) + S_approx.matvec(ns)
  hnss = conjugate_gradient(D, hns, tol=.005, max_iter=10000)[0].detach()
  return hnss

sprhalf = lambda m : conjugate_gradient(D, m, tol=.005, max_iter=10000)[0]

print(f"D - {D}")

r = 80
oversamp = 5

Omega = torch.randn(simPars.nx**2, r + oversamp)
Y = torch.stack([dfpPPH(Omega[:,i]) for i in range(r + oversamp)]).T
Q, R = torch.linalg.qr(Y)
Omega = Omega.to(dtype=torch.float64)
Y = Y.to(dtype=torch.float64)
Q = Q.to(dtype=torch.float64)
T = torch.linalg.solve((Q.T @ Omega).T, (Q.T @ Y).T)
S, Vv = eigsh(T.numpy(),k=r)
S = torch.tensor(S)
print(S)
S[S < 0] = 0
Vv = torch.tensor(Vv, dtype=torch.float64)
V = Q @ Vv
sprV = torch.stack([sprhalf(V[:,i]) for i in range(r)]).T

plt.figure()
im = simPars.plot_std((dspr - (sprV**2)@(S/(S+1)))**.5, plt.gca()) 
plt.colorbar(im, shrink=.6)
plt.title("lrGOGN Standard Deviation",fontsize=24)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-lrgognstd" + ".png"))
plt.close()

fig, ax = plt.subplots(3,3, figsize=(6,5), constrained_layout = True)
ims = []
for i in range(3):
  for j in range(3):
    n = torch.randn_like(m_true, dtype=torch.float64)
    im = simPars.plot_model((mSFGN + sprhalf(V@(((S + 1)**-.5 - 1) * (sprV.T @ n)) + n)).detach(), ax[i,j])
    ims.append(im)
#for im in ims:
#  im.set_clim(vmin=-0.2, vmax=0.2)

fig.colorbar(ims[0], 
  ax=ax,                # span the entire 3x3 grid
  location='right',     # place on the right side
  shrink=0.9,           # slightly shorter than full height to avoid crowding
  pad=0.04,             # spacing from the plots
  aspect=30
)            # controls thinness (higher = thinner)
fig.suptitle("Samples from the lrGOGN Posterior")
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-lrgognpost" + ".png"))
plt.close()


# Gauss Newton Approximation
from sfgn.objectives.utils import gnHessProd

residual = fullResidual(F, d_obs, lambda x: x/sigma_noise)

N = Jhist[0][0].shape[0]
def dfpPPH(n):
  ns = conjugate_gradient(D, n, tol=.005, max_iter=10000)[0].detach()
  hns = gnHessProd(residual, mSFGN, ns)
  hnss = conjugate_gradient(D, hns, tol=.005, max_iter=10000)[0].detach()
  return hnss

sprhalf = lambda m : conjugate_gradient(D, m, tol=.005, max_iter=10000)[0]

print(f"D - {D}")

r = 80
oversamp = 5

Omega = torch.randn(simPars.nx**2, r + oversamp)
Y = torch.stack([dfpPPH(Omega[:,i]) for i in range(r + oversamp)]).T
Q, R = torch.linalg.qr(Y)
T = torch.linalg.solve((Q.T @ Omega).T, (Q.T @ Y).T)
S, Vv = eigsh(T.numpy(),k=r)
S = torch.tensor(S)
print(S)
S[S < 0] = 0
Vv = torch.tensor(Vv)
V = Q @ Vv
sprV = torch.stack([sprhalf(V[:,i]) for i in range(r)]).T

plt.figure()
im = simPars.plot_std((dspr - (sprV**2)@(S/(S+1)))**.5, plt.gca()) 
plt.colorbar(im, shrink=.6)
plt.title("GN Standard Deviation",fontsize=24)
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-gnstd" + ".png"))
plt.close()

fig, ax = plt.subplots(3,3, figsize=(6,5), constrained_layout = True)
ims = []
for i in range(3):
  for j in range(3):
    n = torch.randn_like(m_true)
    im = simPars.plot_model((mSFGN + sprhalf(V@(((S + 1)**-.5 - 1) * (sprV.T @ n)) + n)).detach(), ax[i,j])
    ims.append(im)
#for im in ims:
#  im.set_clim(vmin=-0.2, vmax=0.2)

fig.colorbar(ims[0], 
  ax=ax,                # span the entire 3x3 grid
  location='right',     # place on the right side
  shrink=0.9,           # slightly shorter than full height to avoid crowding
  pad=0.04,             # spacing from the plots
  aspect=30
)            # controls thinness (higher = thinner)
fig.suptitle("Samples from the GN Posterior")
plt.savefig(os.path.join(os.getcwd(), 'figures',  'uq', f"{sigma_noise}-{coverage}-gnpost" + ".png"))
plt.close()



