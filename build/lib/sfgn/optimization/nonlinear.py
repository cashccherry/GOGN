from typing import Callable, Any, List, Tuple
import torch
import time
from sfgn.optimization.utils import line_search_backtrack, line_search_interpolate, lbfgs_invhess, bfgsHistory, conjugate_gradient
from dataclasses import dataclass, field

from sfgn.objectives       import doubleRosenbrock
from sfgn.objectives.utils import jacobian_with_output, parallel_jacobian, gnHessProd

import psutil
import os
import csv

import copy

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB


@dataclass
class optParams: 
  """A class for storing parameters and functionality common different optimization methods"""
  maxiter: int
  maxstep: float
  gtol: float = 0
  ltol: float = 0
  targetx: torch.Tensor | None = None

  def __post_init__(self):
    self.targetx.requires_grad_(False)

  #Get distance of a given x from the targetx, returns the norm of 
  def targetDist(self, x: torch.Tensor):
    if self.targetx is not None:
      assert self.targetx.shape == x.shape
      return float(torch.linalg.norm(self.targetx - x.clone().detach()))
    return float(torch.linalg.norm(x.clone().detach()))
    
@dataclass
class optHist: 
  """A class for storing the optimization history"""
  iteration: int = 0
  loss: List[float]       = field(default_factory=list)
  gradnorm: List[float]   = field(default_factory=list)
  targetdist: List[float] = field(default_factory=list)
  times: List[float]      = field(default_factory=list)
  forwards: List[float]   = field(default_factory=list)
  backwards: List[float]  = field(default_factory=list)

  _start_time: float = 0

  def __post_init__(self):
    self._start_time = time.time()

  def update(self, loss: float, gradnorm: float, targetdist: float, forwards: int, backwards: int):
    self.iteration += 1
    self.loss.append(loss)
    self.gradnorm.append(gradnorm)
    self.targetdist.append(targetdist)
    self.times.append(time.time() - self._start_time)
    self.forwards.append(forwards)
    self.backwards.append(backwards)

  def checkConverged(self, parameters: optParams) -> bool:
    return (self.loss[-1]/self.loss[0] <= parameters.ltol) or (self.gradnorm[-1]/self.gradnorm[0] <= parameters.gtol) or (self.iteration >= parameters.maxiter)

  def writeCSV(self, path: str):
   with open(path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Iteration", "Time", "Forwards", "Backwards", "Loss", "Gradnorm", "Targetdist"])
    for i in range(self.iteration):
      writer.writerow([i,self.times[i],self.forwards[i],self.backwards[i],self.loss[i],self.gradnorm[i],self.targetdist[i]])

  @property
  def report(self) -> str:
    # Define fixed column widths
    label_width = 16
    value_width = 15
    relative_width = 12
    
    reportstr = ""
    reportstr += "╔" + "═" * 58 + "╗\n"
    reportstr += "║ CONVERGENCE REPORT".ljust(59) + "║\n"
    reportstr += "╟" + "─" * 58 + "╢\n"
    reportstr += f"║ • Iteration: {self.iteration:<10}       Time: {self.times[-1]:.6f}             ║\n"
    reportstr += "╟" + "─" * 58 + "╢\n"
    
    # Use consistent formatting with fixed widths
    reportstr += f"║ {'Forwards:':<{label_width}} {self.forwards[-1]:<{value_width}}  Backwards: {self.backwards[-1]:>{relative_width-1}} ║\n"
    reportstr += f"║ {'Loss:':<{label_width}} {self.loss[-1]:<{value_width}.6e}  Relative: {self.loss[-1]/self.loss[0]:>{relative_width}.2e} ║\n"
    reportstr += f"║ {'Gradient Norm:':<{label_width}} {self.gradnorm[-1]:<{value_width}.6e}  Relative: {self.gradnorm[-1]/self.gradnorm[0]:>{relative_width}.2e} ║\n"
    reportstr += f"║ {'Target Distance:':<{label_width}} {self.targetdist[-1]:<{value_width}.6e}  Relative: {self.targetdist[-1]/self.targetdist[0]:>{relative_width}.2e} ║\n"
    reportstr += "╚" + "═" * 58 + "╝\n"
    return reportstr

  @property
  def relativeLoss(self) -> List[float]:
    return [l/self.loss[0] for l in self.loss]

  @property
  def relativeGradnorm(self) -> List[float]:
    return [g/self.gradnorm[0] for g in self.gradnorm]

  @property
  def relativeTargetdist(self) -> List[float]:
    return [t/self.targetdist[0] for t in self.targetlist]
  
  
  
def CG( 
  f: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  smoother: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """Conjugate Gradient Implementation"""
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  loss = f(x)
  grad = torch.autograd.grad(loss,x)[0]
  
  loss = loss.detach()
  grad = smoother(grad.detach())

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)

  gamma = torch.zeros_like(grad)
  gamma.requires_grad_(False)
  
  beta = 0
 
  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    search_direction = -smoother(preconditioner(grad + beta * gamma)) 
    if torch.dot(search_direction, grad) > 0:
      search_direction = -smoother(preconditioner(grad.clone()))
      gamma *= 0

    search_direction.requires_grad_(False)
    gamma = search_direction.clone()

    loss_old = loss
    grad_old = grad

    maxstepi = history.gradnorm[-1] * history.gradnorm[0]**-1 * parameters.maxstep * torch.max(torch.abs(search_direction))**-1

    stepsize, loss, x = line_search(f, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    grad = smoother(grad.detach())

    dot_ggprev = torch.dot(grad_old, grad)
    beta = (torch.dot(grad,grad) - dot_ggprev)/dot_ggprev
    if abs(dot_ggprev) < 1e-10:
      beta = 0
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)
  return x, history
  
def LBFGS( 
  f: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  bfgsMemory: int | None = None,
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  smoother: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """LBFGS Implementation"""
  bfgsHist   = bfgsHistory(m=bfgsMemory)
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  loss = f(x)
  grad = torch.autograd.grad(loss,x)[0]
  
  loss = loss.detach()
  grad = smoother(grad.detach())

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    search_direction = -smoother(lbfgs_invhess(preconditioner(grad), bfgsHist))

    if torch.dot(search_direction, grad) > 0:
      search_direction = -smoother(preconditioner(grad.clone()))

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = history.gradnorm[-1] * history.gradnorm[0]**-1 * parameters.maxstep * torch.max(torch.abs(search_direction))**-1

    stepsize, loss, x = line_search(f, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    grad = smoother(grad.detach())

    bfgsHist.update(stepsize * search_direction, grad-grad_old)
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)
  return x, history, bfgsHist

def LBFGSUQ( 
  f: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  sprhalfinv: Callable[[torch.Tensor], torch.Tensor],
  m0: torch.Tensor | None = None,
  bfgsMemory: int | None = None,
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  box: Tuple[float, float] = (-.1, .1),
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """LBFGS Implementation with Regularization"""
#  def threshold(y):
#    z = y.clone()
#    with torch.no_grad():
#      z[z>box[0]] = box[0]
#      z[z<box[1]] = box[1]
#    return z
  if m0 is None:
    m0 = x0.clone().detach()
  phiHist   = bfgsHistory(m=bfgsMemory)
  bfgsHist   = bfgsHistory(m=bfgsMemory)
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  loss = f(x)
  grad = torch.autograd.grad(loss,x)[0]
  
  loss = loss.detach() + torch.linalg.norm(sprhalfinv(x - m0))**2
  gradphi = grad.detach()
  grad = grad.detach() + sprhalfinv(sprhalfinv(x - m0))

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    search_direction = -preconditioner(lbfgs_invhess(grad, bfgsHist))

    if torch.dot(search_direction, grad) > 0:
      search_direction = -preconditioner(grad.clone())

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
    gradphi_old = gradphi

    maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1
    if torch.isinf(maxstepi):
      print("WTF")
      print(torch.max(torch.abs(grad)))
      maxstepi = 1 
      search_direction = -preconditioner(grad.clone())
      maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1

    stepsize, loss, x = line_search(lambda x : f(x) + torch.linalg.norm(sprhalfinv(x - m0))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    gradphi = grad.detach()
    grad = grad.detach() + sprhalfinv(sprhalfinv(x - m0))

    bfgsHist.update(stepsize * search_direction, grad-grad_old)
    phiHist.update(stepsize * search_direction, gradphi-gradphi_old)
   
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)
  return x, history, bfgsHist, phiHist





import matplotlib.pyplot as plt

def GNS( 
  f: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  regJ: float =  1e-10,
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  smoother: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """GN-S Implementation"""
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  r    = lambda x : (2*f(x) + 1e-15)**.5
  
  Jac, res = jacobian_with_output(r, x)

  misfit = lambda r : .5*torch.sum(r**2)

  loss = .5 * torch.dot(res, res) 
  grad = Jac.T @ res

  loss = loss.detach()
  grad = smoother(grad.detach())

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    Gram             = Jac @ preconditioner(Jac.T) + regJ * torch.eye(Jac.shape[0])
    search_direction = -smoother(preconditioner(Jac.T @ torch.linalg.solve(Gram, res) ))
    if torch.dot(search_direction, grad) > 0:
      print("bad direction")
      search_direction = -smoother(preconditioner(grad.clone()))

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1
 
    stepsize, loss, x = line_search(lambda x : misfit(r(x)), x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
 
    Jac, res = jacobian_with_output(r, x)
  
    loss = .5 * torch.dot(res, res) 
    grad = Jac.T @ res

    loss = loss.detach()
    grad = smoother(grad.detach())
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)
  return x, history 


def GNSpar( 
  fs: List[Callable[[torch.Tensor], torch.Tensor]], 
  x0: torch.Tensor,
  parameters: optParams, 
  regJ: float =  1e-10,
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  smoother: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """GN-S Implementation"""
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  
  Jac, res = parallel_jacobian(fs, x)

  loss = .5 * torch.dot(res, res) 
  grad = Jac.T @ res

  loss = loss.detach()
  grad = smoother(grad.detach())

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), fs[0].forward_counter, fs[0].backward_counter)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    Gram             = Jac @ preconditioner(Jac.T) + regJ * torch.eye(Jac.shape[0])
    search_direction = -smoother(preconditioner(Jac.T @ torch.linalg.solve(Gram, res) ))
    if torch.dot(search_direction, grad) > 0:
      print("bad direction")
      search_direction = -smoother(preconditioner(grad.clone()))

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1
 
    stepsize, loss, x = line_search(lambda x : sum([f(x) + 5e-11 for f in fs]), x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
 
    Jac, res = parallel_jacobian(fs, x)
  
    loss = .5 * torch.dot(res, res) 
    grad = Jac.T @ res

    loss = loss.detach()
    grad = smoother(grad.detach())
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), fs[0].forward_counter, fs[0].backward_counter)
  return x, history 

def GNCG( 
  residual: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  bfgsmemory: int | None = 20,
  eta0: float = .1,
  regJ: float =  1e-10,
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  smoother: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """GN-CG Implementation"""
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  
  res = residual(x)

  loss = .5 * torch.linalg.norm(res)**2 
  grad = torch.autograd.grad(loss,x)[0]

  loss = loss.detach()
  grad = smoother(grad.detach())

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), residual.forward_counter, residual.backward_counter)

  prev_solve_rnorm = 0
  prev_eta         = eta0

  bfgsHist = bfgsHistory(bfgsmemory)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")

    if history.iteration == 1:
      eta = eta0
    else:
      eta = abs(history.gradnorm[-1] - prev_solve_rnorm)/(history.gradnorm[-2])
      if prev_eta**((1 + 5**.5)/2)  > .1:
        eta = max(eta, prev_eta**((1 + 5**.5)/2))
      eta = min(eta0, eta)
    prev_eta = eta

    if print_debug: 
      print(eta)
      print(history.gradnorm[-1]*eta)
      print(torch.linalg.norm(grad)*eta)

    search_direction, prev_solve_rnorm = conjugate_gradient(lambda v : gnHessProd(residual, x, v) + regJ * v, -grad, M=lambda m, hist=copy.deepcopy(bfgsHist) : lbfgs_invhess(m, hist, init=preconditioner), print_debug=print_debug, tol = history.gradnorm[-1]*eta, memory = bfgsHist)
    search_direction = smoother(search_direction)

    print(torch.dot(search_direction, grad))

    if torch.dot(search_direction, grad) > 0:
      print("bad direction")
      search_direction = -smoother(preconditioner(grad.clone()))

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1

    print("Initiating Linesearch")
 
    stepsize, loss, x = line_search(lambda x : .5 * torch.linalg.norm(residual(x))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    grad = smoother(grad.detach())
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), residual.forward_counter, residual.backward_counter)
  return x, history 


def GNCG_dir( 
  residual: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  bfgsmemory: int | None = 20,
  eta0: float = .1,
  regJ: float =  1e-10,
  preconditioner: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  smoother: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> torch.Tensor:
  """GN-CG Implementation"""
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  
  res = residual(x)

  loss = .5 * torch.linalg.norm(res)**2 
  grad = torch.autograd.grad(loss,x)[0]

  loss = loss.detach()
  grad = smoother(grad.detach())

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), residual.forward_counter, residual.backward_counter)

  prev_solve_rnorm = 0
  prev_eta         = eta0

  bfgsHist = bfgsHistory(bfgsmemory)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")

    if history.iteration == 1:
      eta = eta0
    else:
      eta = abs(history.gradnorm[-1] - prev_solve_rnorm)/(history.gradnorm[-2])
      if prev_eta**((1 + 5**.5)/2)  > .1:
        eta = max(eta, prev_eta**((1 + 5**.5)/2))
      eta = min(eta0, eta)
    prev_eta = eta

    if print_debug: 
      print(eta)
      print(history.gradnorm[-1]*eta)
      print(torch.linalg.norm(grad)*eta)

    search_direction, prev_solve_rnorm = conjugate_gradient(lambda v : gnHessProd(residual, x, v) + regJ * v, -grad, M=lambda m, hist=copy.deepcopy(bfgsHist) : lbfgs_invhess(m, hist, init=preconditioner), print_debug=print_debug, tol = history.gradnorm[-1]*eta, memory = bfgsHist)
    search_direction = smoother(search_direction)

    return search_direction

    if torch.dot(search_direction, grad) > 0:
      print("bad direction")
      search_direction = -smoother(preconditioner(grad.clone()))

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1
 
    stepsize, loss, x = line_search(lambda x : .5 * torch.linalg.norm(residual(x))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    grad = smoother(grad.detach())
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), residual.forward_counter, residual.backward_counter)
  return x, history 



 
if __name__ == "__main__":
  #old test of cg
  print("TESTING CG")
  f              = doubleRosenbrock.Sum()
  parameters     = optParams(maxiter = 1000, maxstep=10, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history = CG(f, x0, parameters)
  print(history.report)
  print(xstar)

  print("TESTING LBFGS")
  f              = doubleRosenbrock.Sum()
  parameters     = optParams(maxiter = 1000, maxstep=10, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history, bfgshistory = LBFGS(f, x0, parameters, bfgsMemory = 5)
  print(history.report)
  print(xstar)

  print("TESTING GN-S")
  f              = doubleRosenbrock.Separate()
  parameters     = optParams(maxiter = 10, maxstep=10, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history = GNS(f, x0, parameters, line_search = line_search_backtrack)
  print(history.report)
  print(xstar)

  print("TESTING GN-CG")
  f              = doubleRosenbrock.Separate()
  parameters     = optParams(maxiter = 10, maxstep=10, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history = GNCG(f, x0, parameters, line_search = line_search_backtrack)
  print(history.report)
  print(xstar)


