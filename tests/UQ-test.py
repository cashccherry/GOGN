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
from sfgn.optimization.utils import conjugate_gradient 

def differentiable_fourier_shift(s, shift, dt=1.0):
    """
    Differentiable shift using Fourier transform (circular shift).
    
    Args:
        s: 1D tensor [N]
        shift: scalar tensor (fractional shift in samples)
        dt: sampling interval (optional)
    
    Returns:
        shifted_s: circularly shifted signal
    """
    N = len(s)
    
    # Compute FFT
    S = torch.fft.fft(s)
    
    # Create frequency vector
    k = torch.arange(N, device=s.device)
    
    # Create phase shift factor (differentiable w.r.t shift)
    # For even N, adjust the Nyquist frequency
    if N % 2 == 0:
        k = torch.where(k > N // 2, k - N, k)
    
    # Apply phase shift
    phase_shift = torch.exp(-2j * torch.pi * k * shift / N)
    
    # Shift in frequency domain
    S_shifted = S * phase_shift
    
    # Inverse FFT
    shifted_s = torch.fft.ifft(S_shifted).real
    
    return shifted_s

def deltat(s0,s1):
  ddt = torch.zeros(1) 
  cc = lambda tau : torch.dot(s0, differentiable_fourier_shift(s1, tau)) 
  with torch.no_grad():
    for _ in range(5):
      ddt -= (torch.func.grad(cc)(ddt) / torch.func.jacrev(torch.func.grad(cc))(ddt)).flatten()
  ddt -= (torch.func.grad(cc)(ddt) / torch.func.jacrev(torch.func.grad(cc))(ddt)).flatten()
  return ddt 

def deltat_vectorized(s0, s1):
    """Vectorized version that handles batches"""
    # If inputs are 1D, use original
    if s0.dim() == 1:
        return deltat(s0, s1)
    
    # For batched inputs, flatten, compute, reshape
    original_shape = s0.shape[:-1]
    s0_flat = s0.reshape(-1, s0.shape[-1])
    s1_flat = s1.reshape(-1, s1.shape[-1])
    
    # Compute all time shifts
    results = []
    for b in range(s0_flat.shape[0]):
        results.append(deltat(s0_flat[b], s1_flat[b]))
    
    return torch.stack(results).reshape(original_shape)

coverage = "good25"#sys.argv[1]
sigma_noise = 0#float(sys.argv[2])

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/{coverage}/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

simPars      = SimulationPars(src = src, rec = rec)
m_true       = simPars.load_image_as_velocity_field()
F, d_obs, Fs = simPars.get_F_data(m_true, sigma=sigma_noise)
weights      = simPars.get_geom_weights(d_obs, invPars.kernel)
weights     /= torch.max(weights)


m0  = torch.zeros_like(m_true, requires_grad=True)
d0  = F(m0)
tau = deltat_vectorized(d_obs, d0)
loss = torch.linalg.norm(tau)**2
loss.backward()

fig = plt.figure(figsize=(5.25, 4), constrained_layout = True)
#im = simPars.plot_grad(m0.grad.detach(), plt.gca())
#simPars.plot_src_rec(plt.gca())
print(simPars.rec_loc.shape)
plt.scatter(*tuple(simPars.src_loc[15,0,:]//1000))
simPars.plot_ttpert(tau[15].detach(), plt.gca())

#cbar = plt.colorbar(im)
#cbar.set_label("% Speed Perturbation", fontsize=16)
#cbar.formatter = FuncFormatter(percent_formatter)
#cbar.update_ticks()
plt.legend()
plt.gca().set_aspect('equal')
plt.plot()

#t = torch.arange(d0.shape[-1])*simPars.dt*1.0
#plt.plot(d0[0,0,:])
##plt.plot(d_obs[0,0,:])
#plt.plot(differentiable_fourier_shift(d0[0,0,:], 10))
#
#plt.show()
#
#dt = np.linspace(-.1,.1, 10000)
#plt.plot(dt, [torch.dot(d0[0,0,:], differentiable_fourier_shift(d0[0,0,:], ddt)) for ddt in dt])
#print(dt[np.argmax(torch.tensor([torch.dot(d0[0,0,:], differentiable_fourier_shift(d_obs[0,0,:], ddt)) for ddt in dt]))])
#print(deltat(d0[0,0,:],d0[0,0,:]))
#print(deltat(d0[0,0,:],d_obs[0,0,:]))


#fig,axes = plt.subplots(3,3)
#
#simPars.plot_data(F(0*m_true), axes)
#simPars.plot_data(d_obs, axes)

plt.show()


