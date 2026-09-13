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


coverage = "good" 
sigma_noise = 1e-2

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e4)

fig, axes = plt.subplots(3,4,figsize=(5.25, 4), constrained_layout = True)

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
n_hutch = 5
for _ in range(n_hutch):
  v = torch.randn(m0.shape)
  v = invPars.smooth(v, simPars)
  _, Hv = torch.autograd.functional.vjp(lambda m : fr(m).flatten(), m0, torch.autograd.functional.jvp(lambda m : fr(m).flatten(), m0, v)[1]) 
  prec += n_hutch**-1 * v * Hv

prec[prec<0] = 0

for j, l1 in enumerate([2e3, 8e3, 16e3]):
  mult = 1e-4
  l0   = 5e-1 * 0

#  print(f"eps = f{(mult * l0)**-1/2}")

#  l1   = 10e3
  l2   = 0 * 100e3

  λ1 = mult * l1**2

  print(f'Crossover Wavelength, λ = {λ1:.1e}')

  reg = TikhonovRegularizer(
    Dx = simPars.Dx, Dy = simPars.Dy, 
    Dxt = simPars.Dxt, Dyt = simPars.Dyt, 
    L = simPars.laplacian, 
    λ0 = mult * l0, λ1 = mult * l1**2, λ2 = mult * l1**4, 
    m0 = m0, h = simPars.dx
  )
  print(simPars.dx)

#  im = simPars.plot_grad(prec.detach(), axes[j, 0])
#  print(prec)

  
 
  wavelengths = 2*torch.pi*torch.sqrt(prec/(mult * l1**2) - mult*l0).detach()**-1 / 1e3
 

  im = simPars.plot_std(wavelengths, axes[j,0], cmap='plasma_r')
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
  cbar.set_label("Wavelength [km]", fontsize=12)
  cbar.update_ticks()
  axes[j, 0].set_aspect('equal')
  axes[j, 0].set_title('Crossover Wavelength')


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

  parameters     = optParams(maxiter = 20, maxsim = 100, maxstep=.05, targetx = m_true, ltol = 1e-10, gtol=1e-10)

  print("Running CG")
  f       = waveformMisfit(F, d_obs, weights)
  mCG, historyCG, = CG(f, m0, parameters, reg=reg, h = prec, print_debug=True, line_search = line_search_interpolate_grad)
#  mCG = m0



  im = simPars.plot_model((mCG - m_true).detach(), axes[j, 1])

  axes[j, 1].set_aspect('equal')
  axes[j, 1].set_title(f'Recovery Error, λ = {λ1:.1e}')
  cbar = plt.colorbar(im)
  cbar.formatter = FuncFormatter(percent_formatter)
  cbar.update_ticks()



  im = simPars.plot_model((mCG - m_true).detach(), axes[j, 2])

  axes[j, 2].set_aspect('equal')
  axes[j, 2].set_title(f'with Crossover, λ = {λ1:.1e}')
  cbar = plt.colorbar(im)
  cbar.formatter = FuncFormatter(percent_formatter)
  cbar.update_ticks()


  im = simPars.plot_model(mCG.detach(), axes[j,3], fontsize = 16)
#  simPars.plot_src_rec(plt.gca())
  axes[j, 3].set_aspect('equal')
  axes[j, 3].set_title('Recovery')
  cbar = plt.colorbar(im)
  cbar.set_label("% Speed Perturbation", fontsize=16)
  cbar.formatter = FuncFormatter(percent_formatter)
  cbar.update_ticks()





  im = simPars.plot_hatched_invalid(wavelengths, axes[j, 2], alpha = .1)

  im = simPars.plot_std_contour(wavelengths, axes[j, 2], levels = 10**torch.linspace(-2,3,15), cmap='plasma_r', alpha = 1, linewidths=.5)

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

  axes[j, 2].set_aspect('equal')

  fig.suptitle(f"Tikhonov Regularization with Crossover Wavelengths", fontsize=18)
#  plt.show()
plt.show()
# # 
#os.path.join(os.getcwd(), 'figures',  'acoustic', f"{coverage}-true" + ".png"))
# # #torch.save(m_true, os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-true.pt'))




