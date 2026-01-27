from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple
import torch
import numpy as np
import torch.nn as nn
import deepwave
from deepwave import scalar
from scipy.signal import butter
from scipy.ndimage import gaussian_filter, zoom
from torchaudio.functional import lowpass_biquad
import torchvision
import os
import matplotlib
import matplotlib.pyplot as plt
import math

CWD_file = os.path.dirname(os.path.abspath(__file__))

def biquad(x, sos):
    s = x.shape
    x = x.reshape(s[0]*s[1], s[2])
    # sos: [[b0, b1, b2, a0, a1, a2], ...] for each stage
    for stage in sos:
        b0, b1, b2, a0, a1, a2 = stage
        # Normalize coefficients
        denom = a0
        b0, b1, b2 = b0/denom, b1/denom, b2/denom
        a1, a2 = a1/denom, a2/denom
        
        # Initialize state per batch element
        w1 = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        w2 = torch.zeros_like(w1)
        y = torch.zeros_like(x)
        
        for i in range(x.shape[1]):
            w0 = x[:, i] - a1*w1 - a2*w2
            y[:, i] = b0*w0 + b1*w1 + b2*w2
            w2, w1 = w1.clone(), w0.clone()  # Clone state to avoid in-place
        x = y  # Prepare for next stage
    return x.reshape(s)

@dataclass
class SimulationPars():
  src: torch.Tensor
  rec: torch.Tensor
  nx:   int   = 200
  freq: float = .1
  nt:   int   = 200
  dt:   float = 1
  c0:   float = 3000
  
  def __post_init__(self): 
    self.dx = 480000/self.nx
    self.x = self.dx * torch.arange(self.nx)
    self.X, self.Y         = torch.meshgrid(self.x, self.x, indexing='xy')
    self.peak_time = 1.5/self.freq
 
    self.n_events = self.src.shape[0]
    self.n_recs   = self.rec.shape[0]

    self.src_loc = self.src.reshape(self.n_events, 1, 2)             
    self.rec_loc = self.rec.reshape(1, self.n_recs, 2).repeat(self.n_events, 1, 1)  

    self.source_amplitudes = 1e-5 * (
      deepwave.wavelets.ricker(self.freq, self.nt, self.dt, self.peak_time)
      .repeat(self.n_events, 1, 1)
      )
    
    sos = butter(6, 1.5 * self.freq, fs=1/self.dt, output='sos')
    self.sos = [torch.tensor(sosi).to(torch.float64) for sosi in sos]

  def laplacian(self, u):
    u_grid = u.reshape(self.nx, self.nx)
    laplacian = torch.zeros_like(u_grid)
    
    # Interior points (standard central difference)
    laplacian[1:-1, 1:-1] = (
        u_grid[1:-1, 2:] + u_grid[1:-1, :-2] +  # x neighbors
        u_grid[2:, 1:-1] + u_grid[:-2, 1:-1] -  # y neighbors
        4 * u_grid[1:-1, 1:-1]                  # center
    ) / self.dx**2
    
    # Left boundary (i=0) - forward difference in x
    laplacian[0, 1:-1] = (
        u_grid[0, 2:] + u_grid[0, :-2] +        # x neighbors (central in y)
        u_grid[1, 1:-1] + u_grid[0, 1:-1] -     # forward in x (u[0,j] acts as u[-1,j])
        4 * u_grid[0, 1:-1]
    ) / self.dx**2
    
    # Right boundary (i=-1) - backward difference in x  
    laplacian[-1, 1:-1] = (
        u_grid[-1, 2:] + u_grid[-1, :-2] +      # x neighbors (central in y)
        u_grid[-1, 1:-1] + u_grid[-2, 1:-1] -   # backward in x (u[-1,j] acts as u[n+1,j])
        4 * u_grid[-1, 1:-1]
    ) / self.dx**2
    
    # Bottom boundary (j=0) - forward difference in y
    laplacian[1:-1, 0] = (
        u_grid[2:, 0] + u_grid[:-2, 0] +        # y neighbors (central in x)
        u_grid[1:-1, 1] + u_grid[1:-1, 0] -     # forward in y (u[i,0] acts as u[i,-1])
        4 * u_grid[1:-1, 0]
    ) / self.dx**2
    
    # Top boundary (j=-1) - backward difference in y
    laplacian[1:-1, -1] = (
        u_grid[2:, -1] + u_grid[:-2, -1] +      # y neighbors (central in x)
        u_grid[1:-1, -1] + u_grid[1:-1, -2] -   # backward in y (u[i,-1] acts as u[i,n+1])
        4 * u_grid[1:-1, -1]
    ) / self.dx**2
    
    # Corners - use one-sided differences in both directions
    # Bottom-left (0,0)
    laplacian[0, 0] = (
        u_grid[0, 1] + u_grid[0, 0] +           # forward in x
        u_grid[1, 0] + u_grid[0, 0] -           # forward in y
        4 * u_grid[0, 0]
    ) / self.dx**2
    
    # Bottom-right (-1,0)
    laplacian[-1, 0] = (
        u_grid[-1, 1] + u_grid[-1, 0] +         # backward in x
        u_grid[-1, 0] + u_grid[-2, 0] -         # forward in y
        4 * u_grid[-1, 0]
    ) / self.dx**2
    
    # Top-left (0,-1)
    laplacian[0, -1] = (
        u_grid[0, -1] + u_grid[0, -2] +         # forward in x
        u_grid[1, -1] + u_grid[0, -1] -         # backward in y
        4 * u_grid[0, -1]
    ) / self.dx**2
    
    # Top-right (-1,-1)
    laplacian[-1, -1] = (
        u_grid[-1, -1] + u_grid[-1, -2] +       # backward in x
        u_grid[-1, -1] + u_grid[-2, -1] -       # backward in y
        4 * u_grid[-1, -1]
    ) / self.dx**2
    
    return laplacian.flatten()

  def filt(self,x):
    return biquad(x, self.sos)

  def load_image_as_velocity_field(self, fname: str = "FWI/smiley.png"):
    image = torchvision.io.read_image(os.path.join(CWD_file,fname), mode = torchvision.io.ImageReadMode.GRAY).squeeze().to(torch.float)
    
    image  = torch.flip(torch.tensor(zoom(gaussian_filter(image, 1), self.nx/150, order=1)), (0,))
    image -= image.clone()[0,0]
    c_true = self.c0 + 30 * (image/torch.max(torch.abs(image)))
    m_true = c_true.flatten()/self.c0 - 1
    return m_true * 1.5

  def get_F_data(self, m_true: torch.Tensor, sigma: float = .3**.5) -> Tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    torch.manual_seed(3906)
#    print(self.src_loc.shape)
#    print(self.rec_loc.shape)

   ### Forward Model Function
    F   = lambda m : scalar(self.c0*(1 + m).reshape(self.nx, self.nx), self.dx, self.dt, source_amplitudes=self.source_amplitudes,
                 source_locations  =self.src_loc.flip((-1,))//self.dx,
                 receiver_locations=self.rec_loc.flip((-1,))//self.dx,
                 accuracy=8,
                 pml_freq=self.freq,
                 freq_taper_frac=0.2,
                 time_pad_frac=0.2)[-1]

    Fs = [ lambda m, i=i : scalar(
          self.c0*(1+m).reshape(*2*[self.nx]), self.dx, self.dt,
          source_amplitudes  = self.source_amplitudes[[i]],
          source_locations   = (self.src_loc.flip((-1,))[[i]]//self.dx),
          receiver_locations = (self.rec_loc.flip((-1,))[[i]]//self.dx),
          pml_freq=self.freq, accuracy = 8, freq_taper_frac = .2, time_pad_frac = .2)[-1] 
          for i in range(self.n_events) ]

    ### Get true data
    true_rec_amp = F(m_true)

    d_obs = self.filt(true_rec_amp + sigma * torch.real(1*torch.fft.ifft(torch.abs(torch.fft.fft(true_rec_amp, dim=2)) * torch.randn(true_rec_amp.shape))))
    d_obs.requires_grad_(False)
    return lambda x : self.filt(F(x)), d_obs, [lambda x, j=j : self.filt(Fs[j](x)) for j in range(self.n_events)]
  
  def get_geom_weights(self, d_obs: torch.Tensor, kernel: Callable[[torch.Tensor], torch.Tensor] ) -> torch.Tensor:
    # Geometric Weighting for Observations
    def recdensity(xx, yy):
      recdens = torch.zeros_like(xx)
      for i in range(self.n_recs):
        recdens += self.n_recs**-1 * kernel(((self.rec_loc[0,i,0] - xx)**2 + (self.rec_loc[0,i,1] - yy)**2)**.5)
      return recdens
  
    weights  = torch.linalg.norm(d_obs,dim=2).unsqueeze(-1)**-1
    rdens    = recdensity(self.rec_loc[...,0], self.rec_loc[...,1]).unsqueeze(-1)**-.5
    weights *= rdens/torch.linalg.norm(rdens.flatten(), np.inf)
    weights /= torch.max(weights)
    weights.requires_grad_(False)
    
    return weights

  def plot_data(self, data: torch.Tensor, axes: np.ndarray[matplotlib.axes._axes.Axes], s_inds: List[int] | None = None, r_inds: List[int] | None = None):
    if s_inds is None:
      s_inds = list(range(axes.shape[0]))
    if r_inds is None:
      r_inds = list(range(axes.shape[1]))

    assert axes.shape == (len(s_inds), len(r_inds))

    for i, ri in enumerate(r_inds):
      for j, sj in enumerate(s_inds):
        axes[j,i].plot(self.dt * torch.arange(self.nt), data[sj, ri, :])

    for i, ri in enumerate(r_inds):
      axes[-1, i].set_xlabel(f"Receiver {ri}")
    for j, sj in enumerate(s_inds):
      axes[j, 0].set_ylabel(f"Source {sj}")

  def plot_model(self, m: torch.Tensor, ax: matplotlib.axes._axes.Axes, fontsize: int = 12) -> matplotlib.collections.QuadMesh:
      im = ax.pcolormesh(self.X/1000,self.Y/1000,m.reshape(self.nx, self.nx).cpu(), cmap='RdBu', clim=torch.tensor([-1,1])*.05)
      ax.set_xlabel('x [km]', fontsize=fontsize)
      ax.set_ylabel('z [km]', fontsize=fontsize)
      return im

  def plot_grad(self, m: torch.Tensor, ax: matplotlib.axes._axes.Axes) -> matplotlib.collections.QuadMesh:
      im = ax.pcolormesh(self.X/1000,self.Y/1000,m.reshape(self.nx, self.nx).cpu(), cmap='RdBu')
      ax.set_xlabel('x [km]')
      ax.set_ylabel('z [km]')
      return im

  def plot_std(self, m: torch.Tensor, ax: matplotlib.axes._axes.Axes) -> matplotlib.collections.QuadMesh:
      im = ax.pcolormesh(self.X/1000,self.Y/1000,m.reshape(self.nx, self.nx).cpu(), cmap='turbo')
      ax.set_xlabel('x [km]')
      ax.set_ylabel('z [km]')
      return im

  def plot_src_rec(self, ax: matplotlib.axes._axes.Axes):
    ax.scatter(self.rec_loc[:,:,0].cpu()//1000, self.rec_loc[:,:,1].cpu()//1000, marker = 'v', color = 'y', linewidths=.25, edgecolors='k', s=8, label = 'Receivers')
    ax.scatter(self.src_loc[:,0,0].cpu()//1000, self.src_loc[:,0,1].cpu()//1000, marker = '*', color = 'slategrey', linewidths=.5, edgecolors='k', s=60, label = 'Sources')

  def plot_ttpert(self, tau: torch.Tensor, ax: matplotlib.axes._axes.Axes):
    ax.scatter(self.rec_loc[0,:,0].cpu()//1000, self.rec_loc[0,:,1].cpu()//1000, marker = '.', linewidths=.25, c=tau.flatten(), vmin = -torch.max(torch.abs(tau)), vmax = torch.max(torch.abs(tau)), cmap = 'seismic', s=30, label = 'Receivers')
#    ax.scatter(self.src_loc[:,0,0].cpu()//1000, self.src_loc[:,0,1].cpu()//1000, marker = '*', color = 'slategrey', linewidths=.5, edgecolors='k', s=60, label = 'Sources')
   

@dataclass
class InversionPars():
  sigmak: float    = 60.0e3

  def __post_init__(self):
    self.kernel = lambda r : (self.sigmak**-1 * (2 * torch.pi)**.5) * torch.exp(-r**2/(2*self.sigmak**2))
  
  @property
  def eps(self):
    return 1/(2*self.sigmak**2)

  def smooth(self, v: torch.Tensor, simPars: SimulationPars, background: float = 0, padexp: int = 3, power: float = 1) -> torch.Tensor:
    """This function uses black magic shown to me by Doug Nychka to convolve with the kernel"""
    pad_amount = int(2**(torch.log2(torch.Tensor([simPars.nx]))//1 + padexp).item())
    v = torch.nn.functional.pad(v.reshape(simPars.nx,simPars.nx),[0, pad_amount, 0, pad_amount],mode='constant',value=background)

    with torch.no_grad():
      v_expanded = torch.vstack([v, background*torch.ones_like(v)])
      v_expanded = torch.hstack([v_expanded, background*torch.vstack([torch.ones_like(v), torch.ones_like(v)])])

      X_ext, Y_ext = torch.meshgrid(simPars.dx * torch.arange(simPars.nx + pad_amount), simPars.dx * torch.arange(simPars.nx + pad_amount), indexing='xy')

      X_expanded = torch.vstack([X_ext, X_ext])
      X_expanded = torch.hstack([X_expanded, torch.flip(X_expanded,[1])])
      Y_expanded = torch.vstack([Y_ext, torch.flip(Y_ext,[0,1])])
      Y_expanded = torch.hstack([Y_expanded, torch.flip(Y_expanded,[0,1])])

      kernel_expanded = self.kernel((X_expanded**2 + Y_expanded**2)**.5)
      kernel_expanded /= kernel_expanded.sum()

      v_fft = torch.fft.rfft2(v_expanded)
      kernel_fft = torch.fft.rfft2(kernel_expanded)
    out = torch.fft.irfft2(v_fft * kernel_fft**power)[:simPars.nx,:simPars.nx]

    del v_fft, kernel_fft  # Explicitly delete intermediates
    return out.flatten()

   

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

class fullResidual(nn.Module):
  """This gives a residual F(m) - d"""
  def __init__(self, F: Callable[[torch.Tensor], torch.Tensor], d_obs: torch.Tensor, weights: torch.Tensor):
    super().__init__()
    self.forward_counter  = 0
    self.backward_counter = 0
    self.F                = F
    self.data             = d_obs
    self.weights          = weights 

  def forward(self, x):
    self.forward_counter += 1

    unweighted_res = self.F(x) - self.data
    output         = unweighted_res * self.weights
 
    if x.requires_grad:
      output.register_hook(self._backward_hook)
    return output

  def _backward_hook(self, grad):
    self.backward_counter += 1
    return grad

class waveformMisfit(nn.Module):
  """This gives the full waveform misfit chi(m) = .5 \|F(m) - d\|^2"""
  def __init__(self, F: Callable[[torch.Tensor], torch.Tensor], d_obs: torch.Tensor, weights: torch.Tensor):
    super().__init__()
    self.residual = fullResidual(F, d_obs, weights)

  def forward(self, x):
    outputs = self.residual(x)
    return torch.linalg.norm(outputs.flatten())**2/2
    
  @property
  def forward_counter(self):
    return self.residual.forward_counter

  @property
  def backward_counter(self):
    return self.residual.backward_counter

class sourceMisfits(nn.Module):
  """This gives the GN-S source misfits chi_i(m) = .5 \|F_i(m) - d_i\|^2"""
  def __init__(self, F: Callable[[torch.Tensor], torch.Tensor], d_obs: torch.Tensor, weights: torch.Tensor):
    super().__init__()
    self.forward_counter  = 0
    self.backward_counter = 0
    self.F                = F
    self.data             = d_obs
    self.weights          = weights 

  def forward(self, x):
    self.forward_counter += 1
    d_pred = self.F(x).contiguous()
    weighted_res = (d_pred - self.data) * self.weights
    res_squared = torch.sum(weighted_res**2, dim=[1,2]) / 2
    return res_squared
#    unweighted_res = self.F(x) - self.data
#    output         = unweighted_res * self.weights
# 
#    if x.requires_grad:
#      output.register_hook(self._backward_hook)
#
#    sourcerec = torch.linalg.norm(output, dim=[1,2], keepdim=True)**2/2
#    return sourcerec.squeeze(-1)
    
  def _backward_hook(self, grad):
    self.backward_counter += 1
    return grad


if __name__ == "__main__":
  src      = torch.tensor(np.load(os.path.join(CWD_file, "FWI/good/srcs.npy")))
  rec      = torch.tensor(np.load(os.path.join(CWD_file, "FWI/good/recs.npy")))

  invPars  = InversionPars()

  simPars   = SimulationPars(src = src, rec = rec)
  m_true    = simPars.load_image_as_velocity_field()
  F, d_obs, Fs  = simPars.get_F_data(m_true)
  weights   = simPars.get_geom_weights(d_obs, invPars.kernel)

  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  i      = 0
  print(torch.linalg.norm(F(m_init)[[0]] - Fs[0](m_init)))


  fr  = fullResidual(F, d_obs, weights)
  chi = waveformMisfit(F, d_obs, weights)
  ss  = sourceMisfits(F, d_obs, weights)

  chis = [waveformMisfit(Fs[i], d_obs[[i]], weights[[i]]) for i in range(len(Fs))]

  import time
  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  frout = fr(m_init)
  tf = time.time()-t0
  print(f"Gauss Newton {tf}") 
  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  chiout = chi(m_init)
  tf = time.time()-t0
  print(f"Misfit Comp Time {tf}") 
  t0 = time.time()
  chiout.backward()
  tf = time.time()-t0
  print(f"Misfit Grad Time {tf}") 

  from sfgn.objectives.utils import jacobian_with_output
  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  jac, res = jacobian_with_output(ss, m_init) 
  tf = time.time()-t0
  print(f"Source Subspace Jacobian and Residual Time Module torch.autograd.functional.jacobian with a hook {tf}") 

  F_ss = lambda m : torch.linalg.norm(F(m) - d_obs,dim=[1,2])

  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  jac, res = jacobian_with_output(F_ss, m_init) 
  tf = time.time()-t0
  print(f"Source Subspace Jacobian and Residual Time torch.autograd.functional.jacobian with a hook {tf}") 


  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0  = time.time()
  res = F_ss(m_init)
  Jac = torch.autograd.functional.jacobian(F_ss, m_init) 
  tf  = time.time()-t0
  print(f"Source Subspace Jacobian torch.autograd.functional.jacobian {tf}") 


  import torch
  import time
  import concurrent.futures
  
  # First approach - parallelize individual residual and gradient computations
  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  reslist = []
  Jaclist = []
  
  def compute_residual_grad(i):
      # Need to create a fresh computation graph for each thread
      with torch.no_grad():
          m_local = m_init.detach().clone()
          m_local.requires_grad_()
      
      res = torch.linalg.norm(Fs[i](m_local) - d_obs[[i]])
      res.backward()
      return res, m_local.grad.clone()
  
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results = list(executor.map(compute_residual_grad, range(len(Fs))))
  
  reslist, Jaclist = zip(*results)  # Unpack results
  reslist = list(reslist)
  Jaclist = list(Jaclist)

  print(Jaclist)
  
  tf = time.time()-t0
  print(f"Source Subspace Jacobian and Residual Time, multithreaded {tf}")

  # First approach - parallelize individual residual and gradient computations
  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  reslist = []
  Jaclist = []
  
  def compute_residual_grad(i):
      # Need to create a fresh computation graph for each thread
      with torch.no_grad():
          m_local = m_init.detach().clone()
          m_local.requires_grad_()
      
      res = chis[i](m_local)
      res.backward()
      return res, m_local.grad.clone()
  
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results = list(executor.map(compute_residual_grad, range(len(Fs))))
  
  reslist, Jaclist = zip(*results)  # Unpack results
  reslist = list(reslist)
  Jaclist = list(Jaclist)

  print(Jaclist)
  
  tf = time.time()-t0
  print(f"Source Subspace Jacobian and Residual Time, multithreaded with module {tf}")
  

  
  # Second approach - parallelize forward passes, then combine for backward
  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0 = time.time()
  
  def compute_residual(i):
      return torch.linalg.norm(Fs[i](m_init) - d_obs[[i]])
  
  with concurrent.futures.ThreadPoolExecutor() as executor:
      reslist = list(executor.map(compute_residual, range(len(Fs))))
  
  loss = sum(reslist)
  loss.backward()
  grad = m_init.grad
  
  tf = time.time()-t0
  print(f"Loss and grad, multithreaded with loss as sum of Fs[i](m)s {tf}")



  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0  = time.time()
  reslist = len(Fs)*[None]
  Jaclist = len(Fs)*[None]
  for i in range(len(Fs)):
    reslist[i] = (torch.linalg.norm(Fs[i](m_init) - d_obs[[i]]))
    reslist[i].backward()
    Jaclist[i] = (m_init.grad.clone())
  tf  = time.time()-t0
  print(f"Source Subspace Jacobian and Residual Time, for loop with loss as sum of Fs[i](m)s {tf}") 

  m_init = torch.zeros_like(m_true)
  m_init.requires_grad_()
  t0  = time.time()
  reslist = len(Fs)*[None]
  for i in range(len(Fs)):
    reslist[i] = torch.linalg.norm(Fs[i](m_init) - d_obs[[i]])
  loss = sum(reslist)
  loss.backward()
  grad = m_init.grad
  tf  = time.time()-t0
  print(f"Loss and grad, for loop with loss as sum of Fs[i](m)s {tf}") 


import torch
import time
from loky import get_reusable_executor

def parallel_with_loky_spawn(Fs, d_obs, m_true):
    """
    loky uses spawn by default and handles serialization well
    """
    m_init = torch.zeros_like(m_true)
    t0 = time.time()
    
    def compute_single(i):
        m_local = m_init.detach().clone()
        m_local.requires_grad_()
        res = torch.linalg.norm(Fs[i](m_local) - d_obs[[i]])
        res.backward()
        return res.detach().clone(), m_local.grad.clone()
    
    # loky uses spawn by default and handles lambdas better
    with get_reusable_executor(max_workers=len(Fs)) as executor:
        results = list(executor.map(compute_single, range(len(Fs))))
    
    reslist, Jaclist = zip(*results)
    tf = time.time() - t0
    print(f"Loky (spawn default) - Computation Time: {tf}")
    return list(reslist), list(Jaclist)


if __name__ == "__main__":
  parallel_with_loky_spawn(Fs, d_obs, m_true)

