import os
import sys

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

from sfgn.objectives.FWI import *
from sfgn.optimization.utils import line_search_interpolate_grad, line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *
from sfgn.optimization.utils import conjugate_gradient 

from sfgn.regularization.tikhonov import TikhonovRegularizer


test_name = "epsilonthree"

coverage = "good" 
sigma_noise = 1e-2

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e4)


nx = 200

simPars      = SimulationPars(nx = nx, src = src, rec = rec, freq=.1)

m_true       = simPars.load_image_as_velocity_field()
F, d_obs, Fs = simPars.get_F_data(m_true, sigma=sigma_noise)
weights      = simPars.get_geom_weights(d_obs, invPars.kernel)
weights /= torch.max(weights)

m0        = 0*m_true.clone()
m0.requires_grad_()

chi       = waveformMisfit(F, d_obs, weights)

misfit  = chi(m0)
grad    = torch.autograd.grad(misfit, m0, create_graph=True)[0]
gradsum = torch.sum(grad)
prec    = torch.autograd.grad(gradsum, m0)[0].flatten()
prec   -= torch.min(prec)
prec   += 1e-1


prec = simPars.illumination_map(m = m0)#, weights = weights) # illumination map
fr  = fullResidual(F, d_obs, weights)




prec = torch.zeros_like(m0)
Hv = torch.zeros_like(m0)

n_hutch = 10
for _ in range(n_hutch):
  v = torch.randn(m0.shape)
  v = invPars.smooth(v, simPars)
  _, Hv = torch.autograd.functional.vjp(lambda m : fr(m).flatten(), m0, torch.autograd.functional.jvp(lambda m : fr(m).flatten(), m0, v)[1]) 
  prec += n_hutch**-1 * v * Hv

#prec[prec<0] = 0
prec = torch.abs(prec)

mult = 1e-4


l0   = 5e-1 
l1   = 25e3
l2   = 0 * 100e3

λ0 = mult * l0
λ1 = mult * l1**2
λ2 = mult * l2**4

print(f'Crossover Wavelength, λ = {λ1:.1e}')

reg = TikhonovRegularizer(
  Dx = simPars.Dx, Dy = simPars.Dy, 
  Dxt = simPars.Dxt, Dyt = simPars.Dyt, 
  L = simPars.laplacian, 
  λ0 = mult * l0, λ1 = mult * l1**2, λ2 = mult * l2**4, 
  m0 = m0, h = simPars.dx
)
print(simPars.dx)

diag_A = (
    prec
    + λ0
    + λ1 * (
        2.0 / simPars.dx**2
        + 2.0 / simPars.dx**2
    )
) 


# # im = simPars.plot_std(diag_A, plt.gca())
# # #  simPars.plot_src_rec(plt.gca())
# # #  axes[j, 2].set_aspect('equal')
# # #  axes[j, 2].set_title('10 CG Iterations')
# # lognorm = LogNorm(
# #     vmin=1e-3,# wavelengths.min().item(),
# #     vmax=1e3#   wavelengths.max().item()
# # #       wavelengths.min().item(),
# # #       wavelengths.max().item()   
# # )
# # 
# # print(torch.min(diag_A))
# # print(torch.max(diag_A))
# # print(torch.sum(diag_A))
# # 
# # im.set_norm(lognorm)
# # 
# # cbar = plt.colorbar(im)
# # cbar.set_label("Ratio", fontsize=12)
# # cbar.update_ticks()
# # plt.show()




r = lambda ξ : prec/(λ0 + λ1 * ξ **2 + λ2 * ξ **4)

fig, axes = plt.subplots(1,4,figsize=(14.45, 3), constrained_layout = True)

for j, ξ in enumerate([2*torch.pi/1e3, 2*torch.pi/10e3, 2*torch.pi/100e3, 2*torch.pi/1000e3]):
  im = simPars.plot_std(r(ξ), axes[j], cmap='seismic')
  #  simPars.plot_src_rec(plt.gca())
  #  axes[j, 2].set_aspect('equal')
  #  axes[j, 2].set_title('10 CG Iterations')
  lognorm = LogNorm(
      vmin=1e-3,# wavelengths.min().item(),
      vmax=1e3#   wavelengths.max().item()
  #       wavelengths.min().item(),
  #       wavelengths.max().item()   
  )
  
  im.set_norm(lognorm)
  
  cbar = plt.colorbar(im)
  cbar.set_label("Ratio", fontsize=12)
  cbar.update_ticks()
  axes[j].set_aspect('equal')
  axes[j].set_title(f'Wavelength = {2*torch.pi/(ξ*1e3):.1f} km')

fig.suptitle("Data-to-Regularization Ratio", fontsize=24)
plt.savefig(f"{test_name}_crossover.png")

plt.close('all')


fig, axes = plt.subplots(1,4,figsize=(14.45, 3), constrained_layout = True)
wavelengths = 2*torch.pi*torch.sqrt(prec/λ1 - λ0).detach()**-1 / 1e3

parameters     = optParams(maxiter = 40, maxsim = 100, maxstep=.05, targetx = m_true, ltol = 1e-10, gtol=1e-10)

print("Running CG")
f       = waveformMisfit(F, d_obs, weights)
mCG, historyCG, = CG(f, m0, parameters, reg=reg, h = prec, print_debug=True, line_search = line_search_interpolate_grad)
#  mCG = m0

im = simPars.plot_model((m_true).detach(), axes[0])

axes[0].set_aspect('equal')
axes[0].set_title(f'Target Model')
cbar = plt.colorbar(im)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()


im = simPars.plot_model((mCG).detach(), axes[1])

axes[1].set_aspect('equal')
axes[1].set_title(f'Recovery')
cbar = plt.colorbar(im)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()





# # im = simPars.plot_std(wavelengths, axes[j,0], cmap='plasma_r')
# # #  simPars.plot_src_rec(plt.gca())
# # #  axes[j, 2].set_aspect('equal')
# # #  axes[j, 2].set_title('10 CG Iterations')
# # lognorm = LogNorm(
# #     vmin=1e-3,# wavelengths.min().item(),
# #     vmax=1e3#   wavelengths.max().item()
# # #       wavelengths.min().item(),
# # #       wavelengths.max().item()   
# # )
# # 
# # im.set_norm(lognorm)
# # 
# # cbar = plt.colorbar(im)
# # cbar.set_label("Wavelength [km]", fontsize=12)
# # cbar.update_ticks()
# # axes[j, 0].set_aspect('equal')
# # axes[j, 0].set_title('Crossover Wavelength')


#  n = grad.detach()
#
#  print(torch.max(n))
#  
#  im = simPars.plot_grad(n.detach(), axes[j, 1])
#  axes[j, 1].set_aspect('equal')
#  axes[j, 1].set_title('Gradient')
#  cbar = plt.colorbar(im)
#  cbar.formatter = FuncFormatter(percent_formatter)
#  cbar.update_ticks()
#  
#  im = simPars.plot_grad(conjugate_gradient(lambda m : reg.hess_vec(m0, m) + prec * m, n, tol=.005, max_iter=10000)[0].detach(), axes[j, 1])



im = simPars.plot_model((mCG - m_true).detach(), axes[2])

axes[2].set_aspect('equal')
axes[2].set_title(f'Recovery Error')
cbar = plt.colorbar(im)
cbar.formatter = FuncFormatter(percent_formatter)
cbar.update_ticks()



# im = simPars.plot_model((mCG - m_true).detach(), axes[3])
# 
# axes[3].set_aspect('equal')
# axes[3].set_title(f'Error with Crossover')
# cbar = plt.colorbar(im)
# cbar.formatter = FuncFormatter(percent_formatter)
# cbar.update_ticks()




im = simPars.plot_hatched_invalid(wavelengths, axes[3], alpha = .1)
#, levels = 10**torch.linspace(-2,3,15)
im = simPars.plot_std(wavelengths, axes[3], cmap='plasma_r', alpha = 1, linewidths=.5)

lognorm = LogNorm(
    vmin=1e-3,# wavelengths.min().item(),
    vmax=1e3#   wavelengths.max().item()
#       wavelengths.min().item(),
#       wavelengths.max().item()   
)

im.set_norm(lognorm)

sm = plt.cm.ScalarMappable(cmap='plasma_r', norm = lognorm)


#  simPars.plot_src_rec(axes[j, 0])
cbar = plt.colorbar(im)
cax = cbar.ax
cbar = plt.colorbar(sm, cax=cax)

cbar.set_label("Crossover Wavelength [km]", fontsize=12)
cbar.mappable = sm

cbar.ax.set_yscale('log') 

axes[3].set_aspect('equal')



fig.suptitle(f"Inversion Error with Crossover Wavelengths", fontsize=18)

plt.savefig(f"{test_name}_error.png")
plt.close('all')

# # 
#os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-true" + ".png"))
# # #torch.save(m_true, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-true.pt'))

fig, axes = plt.subplots(1,4,figsize=(14.45, 3), constrained_layout = True)

ε_grad = mCG 
λ_grad = (simPars.Dxt(simPars.Dx(mCG)) + simPars.Dyt(simPars.Dy(mCG))) 

print(torch.linalg.norm(ε_grad))
print(λ_grad)

GNprod = lambda v : torch.autograd.functional.vjp(lambda m : fr(m).flatten(), mCG, torch.autograd.functional.jvp(lambda m : fr(m).flatten(), mCG, v)[1])[1]

def A(m):
    return reg.hess_vec(mCG, m) + prec * m
# Get eigenvalue bounds properly
v_max = grad.clone() 

for i in range(10):
  v_max  = A(v_max)
  v_max /= torch.linalg.norm(v_max)
λ_max = torch.dot(v_max, A(v_max))
omega = .9 * 2/λ_max


#def preconditioner(g, n_iter=30):
#    u = torch.zeros_like(g)
#
#    diag_A = (
#        prec
#        + λ0
#        + λ1 * (
#            2.0 / simPars.dx**2
#            + 2.0 / simPars.dx**2
#        )
#    ) 
#
#    omega = 0.01
#
#    for _ in range(n_iter):
#        r = g - A(u)
#        print(
#            i,
#            "||u|| =", torch.linalg.norm(u).item(),
#            "||r|| =", torch.linalg.norm(r).item(),
#            "max|u| =", torch.max(torch.abs(u)).item(),
#            "max|r| =", torch.max(torch.abs(r)).item(),
#        )
#        u = u + omega * r / diag_A
#
#    return u

@torch.no_grad()
def preconditioner(g):
    u = torch.zeros_like(g)
    for _ in range(5000):
        r = g - A(u)  # Residual
        u = u + omega * r
    return u

with torch.no_grad():
  dm_dε_GN   = conjugate_gradient(lambda m : reg.hess_vec(mCG, m) + GNprod(m), -ε_grad, M=preconditioner, tol=1e-2*torch.linalg.norm(ε_grad), max_iter=100, print_debug=True)[0]
  dm_dλ_GN   = conjugate_gradient(lambda m : reg.hess_vec(mCG, m) + GNprod(m), -λ_grad, M=preconditioner, tol=1e-2*torch.linalg.norm(λ_grad), max_iter=100, print_debug=True)[0]
  
  print(dm_dλ_GN)
  
  dm_dε_prec = conjugate_gradient(lambda m : reg.hess_vec(mCG, m) + prec * m, -ε_grad, tol=1e-6*torch.linalg.norm(ε_grad), max_iter=10000, print_debug=True)[0]
  dm_dλ_prec = conjugate_gradient(lambda m : reg.hess_vec(mCG, m) + prec * m, -λ_grad, tol=1e-6*torch.linalg.norm(λ_grad), max_iter=10000, print_debug=True)[0]
  
  print(dm_dλ_prec)

maximum_ε = max(
 torch.linalg.norm(dm_dε_GN,    ord= torch.inf), 
 torch.linalg.norm(dm_dε_prec,  ord= torch.inf), 
)

maximum_λ = max(
 torch.linalg.norm(dm_dλ_GN,    ord= torch.inf), 
 torch.linalg.norm(dm_dλ_prec,  ord= torch.inf) 
)

im = simPars.plot_grad((dm_dε_GN).detach(), axes[0], cmap='turbo')#, clim = torch.Tensor([-1,1]) * maximum_ε)
cbar = plt.colorbar(im)
cbar.set_label("Sensitivity", fontsize=12)

axes[0].set_aspect('equal')
axes[0].set_title(f'GN ε Sensitivity')

im = simPars.plot_grad((dm_dλ_GN).detach(), axes[1], cmap='turbo')#, clim = torch.Tensor([-1,1]) * maximum_λ)
cbar = plt.colorbar(im)
cbar.set_label("Sensitivity", fontsize=12)

axes[1].set_aspect('equal')
axes[1].set_title(f'GN λ Sensitivity')

im = simPars.plot_grad((dm_dε_prec).detach(), axes[2], cmap='turbo')#, clim = torch.Tensor([-1,1]) * maximum_ε)
cbar = plt.colorbar(im)
cbar.set_label("Sensitivity", fontsize=12)

axes[2].set_aspect('equal')
axes[2].set_title(f'Prec ε Sensitivity')

im = simPars.plot_grad((dm_dλ_prec).detach(), axes[3], cmap='turbo')#, clim = torch.Tensor([-1,1]) * maximum_λ)
cbar = plt.colorbar(im)
cbar.set_label("Sensitivity", fontsize=12)

axes[3].set_aspect('equal')
axes[3].set_title(f'Prec λ Sensitivity')

fig.suptitle("Regularization Sensitivity Analysis", fontsize=24)

plt.savefig(f"{test_name}_sensitivity.png")
plt.close('all')



