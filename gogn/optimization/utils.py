from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, Any
import numpy as np
import torch


@dataclass
class bfgsHistory:
  m: int | None         = None
  S: List[torch.Tensor] = field(default_factory=list)
  Y: List[torch.Tensor] = field(default_factory=list)
  
  def update(self, s: torch.Tensor, y: torch.Tensor):
    """Update history"""
    self.S.append(s.detach().clone())
    self.Y.append(y.detach().clone())

    assert len(self.Y) == len(self.S) #make sure no funny business
    
    if self.m is not None:
      if len(self.Y) > self.m: #don't hold onto more memory than necessary
        del self.S[0]
        del self.Y[0]

  def __call__(self):
    return zip(self.S, self.Y)

  def reversed(self):
    return zip(reversed(self.S), reversed(self.Y))

def lbfgs_invhess(
    q:       torch.Tensor,
    history: bfgsHistory,
    init:    Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    scale:   bool  = True,
    tikhReg: float = 0
):
  q = torch.clone(q)

  alpha = []
  rho   = []

  for s,y in history.reversed():  
    rho.append(torch.dot(y + tikhReg*s,s)**-1)
    alpha.append(rho[-1] * torch.dot(s,q))
    q = q - alpha[-1] * (y + tikhReg*s)

  q = init(q) 

  alpha.reverse()
  rho.reverse()

  if scale and len(history.Y) > 0:
    dot_ys = torch.dot(history.Y[-1], history.S[-1])
    dot_yy = torch.dot(history.Y[-1], history.Y[-1])
    if dot_yy > 1e-10:
      q *= dot_ys/dot_yy

  for (s, y), alphai, rhoi in zip(history(), alpha, rho):
    beta = rhoi * torch.dot(y + tikhReg*s, q)
    q = q + s*(alphai - beta)

  return q

def line_search_backtrack(
    f:            Callable[[torch.Tensor], torch.Tensor], 
    x:            torch.Tensor, 
    p:            torch.Tensor, 
    current_loss: float, 
    phiprime: float | None = None,
    max_step:     float = 1.0, 
    max_evals:    int   = 10,
    print_debug:  bool  = True 
) -> Tuple[float, torch.Tensor, torch.Tensor]:
  """
  NEVER RECOMPUTES - stores the last successful computation.
  """
  steps = [0]
  losses = [current_loss]
  step = max_step 
  
  # Store the last successful computation
  last_successful_step = step
  last_successful_x_new = None
  last_successful_loss_new = None
  
  for i in range(max_evals):
    # Compute once
    x_new = x + step * p
    x_new.requires_grad_(True)
    loss_new = f(x_new)
    
    loss_val = loss_new.item()
    steps.append(float(step))
    losses.append(loss_val)
    
    # Store if successful
    if loss_val < current_loss:
      last_successful_step = step
      last_successful_x_new = x_new
      last_successful_loss_new = loss_new
    
    # Check if we should return
    if (loss_val < current_loss or i == max_evals - 1):
      if print_debug:
        print("SUCCESSFUL LINESEARCH:")
        print(f"step : {steps}")
        print(f"loss : {losses}")
      
      # Return stored values - NO RECOMPUTATION
      if last_successful_x_new is not None:
        return last_successful_step, last_successful_loss_new, last_successful_x_new
      else:
        return step, loss_new, x_new
    
    # Update step size (same as before)
    step = steps[-1] / 2
      
  if print_debug:
    print("FAILED LINESEARCH")
    print(f"step : {steps}")
    print(f"phi   : {losses}")
  
  # Return stored values - NO RECOMPUTATION
  if last_successful_x_new is not None:
    return last_successful_step, last_successful_loss_new, last_successful_x_new
  else:
    return step, loss_new, x_new

def line_search_interpolate_grad(
    f:            Callable[[torch.Tensor], torch.Tensor], 
    x:            torch.Tensor, 
    p:            torch.Tensor, 
    current_loss: float, 
    phiprime:     float | None = None,
    max_step:     float = 1.0, 
    max_evals:    int   = 10,
    print_debug:  bool  = True 
) -> Tuple[float, torch.Tensor, torch.Tensor]:
  """
  NEVER RECOMPUTES - stores the last successful computation.
  """
  def current_state():
    x_current = x.detach().clone().requires_grad_(True)
    return 0.0, f(x_current), x_current

  steps = [0]
  losses = [current_loss]
  step = max_step
  
  # Store the last successful computation
  last_successful_step = step
  last_successful_x_new = None
  last_successful_loss_new = None
  
  for i in range(max_evals):
    if not np.isfinite(float(step)) or float(step) < 0:
      step = steps[-1] / 2 if len(steps) > 1 else max_step / 2

    # Compute once
    x_new = x + step * p
    x_new.requires_grad_(True)
    loss_new = f(x_new)
    
    loss_val = loss_new.item()
    steps.append(float(step))
    losses.append(loss_val)
    finite_loss = np.isfinite(loss_val)
    
    # Store if successful
    if finite_loss and loss_val < current_loss:
      last_successful_step = step
      last_successful_x_new = x_new
      last_successful_loss_new = loss_new

    # Check if we should return
    if ((finite_loss and loss_val < losses[0]) or i == max_evals - 1) and len(steps) > 1:
      if last_successful_x_new is not None:
          if print_debug:
            print("SUCCESSFUL LINESEARCH:")
            print(f"step : {steps}")
            print(f"loss : {losses}")
          return last_successful_step, last_successful_loss_new, last_successful_x_new
      else:
          if print_debug:
            print("FAILED LINESEARCH")
            print(f"step : {steps}")
            print(f"loss : {losses}")
            print("No finite decreasing step found; keeping current iterate")
          return current_state()
    
    # Update step size (same as before)
    if finite_loss and (len(steps) >= 2 and len(steps) < 6):
      phipp = 2 * (step**-2 * (losses[-1] - losses[0]) - step**-1 * phiprime)
      step = - phiprime/phipp
      step = min(max_step, max(step, 0))
    else:
      step = steps[-1] / 2
    
    if step == 0:
      step = steps[-1] / 2
  
  if print_debug:
    print("FAILED LINESEARCH")
    print(f"step : {steps}")
    print(f"phi   : {losses}")
  
  # Return stored values - NO RECOMPUTATION
  if last_successful_x_new is not None:
    return last_successful_step, last_successful_loss_new, last_successful_x_new
  else:
    return current_state()


def line_search_interpolate(
    f:            Callable[[torch.Tensor], torch.Tensor], 
    x:            torch.Tensor, 
    p:            torch.Tensor, 
    current_loss: float, 
    phiprime: float | None = None,
    max_step:     float = 1.0, 
    max_evals:    int   = 10,
    print_debug:  bool  = True 
) -> Tuple[float, torch.Tensor, torch.Tensor]:
  """
  NEVER RECOMPUTES - stores the last successful computation.
  """
  steps = [0]
  losses = [current_loss]
  step = max_step / 15
  
  # Store the last successful computation
  last_successful_step = step
  last_successful_x_new = None
  last_successful_loss_new = None
  
  for i in range(max_evals):
    # Compute once
    x_new = x + step * p
    x_new.requires_grad_(True)
    loss_new = f(x_new)
    
    loss_val = loss_new.item()
    steps.append(float(step))
    losses.append(loss_val)
    
    # Store if successful
    if loss_val < current_loss:
      last_successful_step = step
      last_successful_x_new = x_new
      last_successful_loss_new = loss_new
    
    # Check if we should return
    if (loss_val < current_loss or i == max_evals - 1) and len(steps) > 3:
      if print_debug:
        print("SUCCESSFUL LINESEARCH:")
        print(f"step : {steps}")
        print(f"loss : {losses}")
      
      # Return stored values - NO RECOMPUTATION
      if last_successful_x_new is not None:
          return last_successful_step, last_successful_loss_new, last_successful_x_new
      else:
          return step, loss_new, x_new
    
    # Update step size (same as before)
    if len(steps) == 2:
      step = max_step / 10
    elif len(steps) == 3:
      step = quadratic_interpmin(steps, losses)
      step = min(max_step, max(step, 0))
    else:
      step = steps[-1] / 2
    
    if step == 0:
      step = steps[-1] / 2
  
  if print_debug:
    print("FAILED LINESEARCH")
    print(f"step : {steps}")
    print(f"phi   : {losses}")
  
  # Return stored values - NO RECOMPUTATION
  if last_successful_x_new is not None:
    return last_successful_step, last_successful_loss_new, last_successful_x_new
  else:
    return step, loss_new, x_new

def quadratic_interpmin(x: list[float], y: list [float]) -> float:
  coeffs = np.polyfit(torch.tensor(x), torch.tensor(y), 2)
  return -float(coeffs[1] / (2 * coeffs[0])) if coeffs[0] > 0 else float(x[-1] / 2)

def conjugate_gradient(
    A:            Callable[[torch.Tensor], torch.Tensor], 
    b:            torch.Tensor, 
    M:            Callable[[torch.Tensor], torch.Tensor] = lambda x : x.clone(),
    max_iter:     int                = 100, 
    min_iter:     int                = 0,
    tol:          float              = 1e-6, 
    memory:       bfgsHistory | None = None,
    print_debug:  bool               = False 
) -> Tuple[torch.Tensor, float]:
  """
  Preconditioned Conjugate Gradient method
  
  Parameters:
  A:            Linear operator
  b:            RHS vector
  max_iter:     Maximum number of iterations
  min_iter:     Minimum number of iterations
  tol:          Convergence tolerance
  memory:       BFGS memory to store preconditioner

  Returns:
  x:     Solution vector
  rnorm: Residual norm
  """  
  if memory is not None:
    if memory.m is not None:
      memory.m += memory.m % 2
  inds = []
  x    = torch.zeros_like(b)
  r    = -b.clone()
  
  z = M(r)
  if not torch.isfinite(z).all():
    if print_debug:
      print("Non-finite preconditioned residual detected in Conjugate Gradient")
    return x, float("inf")
  p = -z.clone()
  
  cycle = 1
  for j in range(max_iter):
    rsq   = float(torch.dot(r, z))
    rnorm = float(torch.linalg.norm(r))

    if print_debug:
      print(f"CG iter{j} - residual norm {rnorm/tol}")
    if rnorm <= tol and j >= min_iter:
      break

    Ap = A(p)
    pAp = torch.dot(p, Ap)

    if (not np.isfinite(rsq)) or (not torch.isfinite(Ap).all()) or (not torch.isfinite(pAp)):
      if print_debug:
        print("Non-finite CG quantity detected; returning current iterate")
      break

    if pAp < -1e-7 * torch.dot(p, p):
      print(f"Negative Curvature Detected in Conjugate Gradient :  {pAp / torch.dot(p,p)}")
      break

    if memory is not None:
      if memory.m is None:
        memory.update(p, Ap)
        inds.append(j)
      else: 
        if j < memory.m:
          memory.update(p, Ap)
          inds.append(j)
        elif (j * 2**-cycle + 1 - memory.m / 2).is_integer() and (j * 2**-cycle + 1 - memory.m / 2 <= memory.m / 2):
          l        = int(j * 2**-cycle + 1 - memory.m / 2)
          leaveind = inds.index((2*l - 1) * 2**(cycle - 1))
          if print_debug:
            print("l - ", l, " -- leavind - ", leaveind, "len - ", len(memory.Y))
          del inds[leaveind]
          memory.update(p, Ap)
          inds.append(j)
          if l == int(memory.m/2):
            cycle += 1

    alpha = rsq / pAp
    x    += alpha * p
    r    += alpha * Ap 

    z = M(r)
    if not torch.isfinite(z).all():
      if print_debug:
        print("Non-finite preconditioned residual detected in Conjugate Gradient")
      break

    beta = torch.dot(r,z) / rsq
    p = -z + beta * p 
  print("CG DONE") 
  return x, rnorm


import torch
