from typing import Callable, Any, List, Tuple
import torch
import time
from sfgn.optimization.utils import line_search_backtrack, line_search_interpolate, lbfgs_invhess, bfgsHistory, conjugate_gradient
from dataclasses import dataclass, field
from scipy.sparse.linalg import LinearOperator, eigsh

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
  D: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  h: torch.Tensor | None = None,
  m0: torch.Tensor | None = None,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """Conjugate Gradient Implementation with Regularization"""
  if h is None:
    h = torch.ones_like(x0) 

  if m0 is None:
    m0 = x0.clone().detach()

  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  loss = f(x)
  grad = torch.autograd.grad(loss,x)[0]
  
  loss = loss.detach()
  grad = grad.detach()

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)

  gamma = torch.zeros_like(grad)
  gamma.requires_grad_(False)
  
  beta = 0
 
  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    with torch.no_grad():
      search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad + beta * gamma, tol=.005, max_iter=10000)[0]

    if torch.dot(search_direction, grad) > 0:
      with torch.no_grad():
        search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad.clone(), tol=.005, max_iter=10000)[0]
      gamma *= 0

    search_direction.requires_grad_(False)
    gamma = search_direction.clone()

    loss_old = loss
    grad_old = grad

    maxstepi = history.gradnorm[-1] * history.gradnorm[0]**-1 * parameters.maxstep * torch.max(torch.abs(search_direction))**-1

    stepsize, loss, x = line_search(lambda x : f(x) + .5 * torch.linalg.norm(D(x - m0))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    grad = grad.detach()

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
  D: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  h: torch.Tensor | None = None,
  m0: torch.Tensor | None = None,
  bfgsMemory: int | None = None,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """LBFGS Implementation with Regularization"""
  if h is None:
    h = torch.ones_like(x0) 

  if m0 is None:
    m0 = x0.clone().detach()

  phiHist   = bfgsHistory(m=bfgsMemory)
  bfgsHist   = bfgsHistory(m=bfgsMemory)
  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  loss = f(x)
  grad = torch.autograd.grad(loss,x)[0]
  
  loss = loss.detach() + torch.linalg.norm(D(x - m0))**2
  gradphi = grad.detach()
  grad = grad.detach() + D(D(x - m0))

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")
    with torch.no_grad():
      search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, lbfgs_invhess(grad, bfgsHist), tol=.005, max_iter=10000)[0]

    if torch.dot(search_direction, grad) > 0:
      with torch.no_grad():
        search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad.clone(), tol=.005, max_iter=10000)[0]

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
    gradphi_old = gradphi

    maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1
    if torch.isinf(maxstepi):
      print("How is this possible")
      print(torch.max(torch.abs(grad)))
      maxstepi = 1 
      with torch.no_grad():
        search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad.clone(), tol=.005, max_iter=10000)[0]
      maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1

    stepsize, loss, x = line_search(lambda x : f(x) + .5 * torch.linalg.norm(D(x - m0))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    gradphi = grad.detach()
    grad = grad.detach() + D(D(x - m0))

    bfgsHist.update(stepsize * search_direction, grad-grad_old)
    phiHist.update(stepsize * search_direction, gradphi-gradphi_old)
   
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), f.forward_counter, f.backward_counter)
  return x, history, bfgsHist, phiHist

def SFGN( 
  fs: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  D: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  h: torch.Tensor | None = None,
  m0: torch.Tensor | None = None,
  cgtol: float = 1e-3,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True 
  ) -> Tuple[torch.Tensor, optHist]:
  """SFGN Implementation"""
  if h is None:
    h = torch.ones_like(x0) 

  if m0 is None:
    m0 = x0.clone().detach()

  prec = conjugate_gradient(lambda m : D(D(m)), h, tol=.005, max_iter=10000)[0]

  #def preconditioner(g):
  #  with torch.no_grad():
  #    out = h**-1 * g#conjugate_gradient(lambda m : D(D(m)) + h * m, g.clone(), tol=.005, max_iter=100)[0]
  #  return out

  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()

  Jac, res = parallel_jacobian(fs, x)

  loss = .5 * torch.linalg.norm(res)**2 + .5 * torch.linalg.norm(D(x.clone() - m0))**2
  grad = Jac.T @ res  

  loss = loss.detach()
  grad = grad.detach()

  history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), fs[0].forward_counter, fs[0].backward_counter)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")

    if print_debug: 
      print(history.gradnorm[-1])
      print(torch.linalg.norm(grad))


#   Jacadj = torch.stack([conjugate_gradient(lambda v : D(D(v)), Jac[i,:], tol=cgtol, max_iter=1000)[0] for i in range(Jac.shape[0])]).T
    #Jacadj = Jac.T
    #a = x.clone().detach() - m0 + Jacadj @ res
    #M = Jac @ Jacadj * sigma**-2
    #y = torch.linalg.solve(torch.eye(Jac.shape[0]) + M, Jac @ a)
#      search_direction = - ( a - Jacadj @ y) 
    #search_direction = - conjugate_gradient(lambda v : D(D(v)), Jac.T @ res - Jac.T @ y, tol=cgtol, max_iter=1000)[0] + m0 - x.clone().detach()
    #kappa = torch.mean(D(D(torch.ones(Jac.shape[1]))))
    #G = Jac @ Jac.T
    #G2inv = torch.linalg.pinv(G @ G )
    #def preconditioner(g):
    #  return  Jac.T @ (G2inv @ (Jac @ g)) + kappa*-1 * g
    search_direction, prev_solve_rnorm = conjugate_gradient(lambda v : Jac.T@(Jac@v) + D(D(v)), -grad - D(x.clone().detach() - m0), M=lambda m : prec**-1 * m, tol = cgtol, max_iter=10000)

    print(torch.dot(search_direction, grad))

    if torch.dot(search_direction, grad) > 0:
      print("bad direction")
      search_direction = -grad.clone()

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1

    print("Initiating Linesearch")
 
    stepsize, loss, x = line_search(lambda x : sum([f(x) + 5e-11 for f in fs]) + .5 * torch.linalg.norm(D(x.clone() - m0))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)

    Jac, res = parallel_jacobian(fs, x)

    loss = loss.detach()
    grad = Jac.T @ res 
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), fs[0].forward_counter, fs[0].backward_counter)
  return x, history 



def GNCG( 
  residual: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  D: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  h: torch.Tensor | None = None,
  m0: torch.Tensor | None = None,
  bfgsmemory: int | None = 20,
  maxcgiter: int | None = 40,
  eta0: float = .1,
  line_search: Callable[
    [Callable[[torch.Tensor], torch.Tensor], 
     torch.Tensor, torch.Tensor, 
     float, float,  int, bool], 
    Tuple[float, torch.Tensor, torch.Tensor]] = line_search_interpolate,
  print_debug: bool = True
  ) -> Tuple[torch.Tensor, optHist]:
  """GN-CG Implementation"""
  if h is None:
    h = torch.ones_like(x0) 

  if m0 is None:
    m0 = x0.clone().detach()

  n = len(h)
  
  def A(x):
      return D(D(x)) + h * x
  
  # Get eigenvalue bounds properly
  def numpy_matvec(x_np):
      return A(torch.tensor(x_np)).numpy()
  
  A_op = LinearOperator((n, n), matvec=numpy_matvec)
  λ_min, λ_max = eigsh(A_op, k=2, which='BE', return_eigenvectors=False, tol=1e-3)
  
  # Ensure positive and properly scaled
  if λ_min <= 0:
      raise ValueError(f"Matrix not SPD: λ_min = {λ_min}")
  
  # Scale A so eigenvalues are in [-1, 1]
  # A_scaled = (2A - (λ_max+λ_min)I)/(λ_max-λ_min)
  omega = .9 * 2/λ_max
  
  def A(x):
      return D(D(x)) + h * x
  
  # Choose omega conservatively to ensure convergence
  # For SPD A, convergence if 0 < ω < 2/λ_max
  
  def preconditioner(g):
      u = torch.zeros_like(g)
      for _ in range(250):
          r = g - A(u)  # Residual
          u = u + omega * r
      return u

  print(preconditioner(torch.randn(len(h))))

  history    = optHist()
  x          = x0.clone()
  x.requires_grad_()
  
  res = residual(x)

  loss = .5 * torch.linalg.norm(res)**2 + .5 * torch.linalg.norm(D(x.clone() - m0))**2
  grad = torch.autograd.grad(loss,x)[0]

  loss = loss.detach()
  grad = grad.detach()

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

    search_direction, prev_solve_rnorm = conjugate_gradient(lambda v : gnHessProd(residual, x, v) + D(D(v)), -grad, M=lambda m, hist=copy.deepcopy(bfgsHist) : lbfgs_invhess(m, hist, init=preconditioner), print_debug=print_debug, tol = history.gradnorm[-1]*eta, memory = bfgsHist, max_iter=maxcgiter)
    #search_direction = Dinv(search_direction)

    print(torch.dot(search_direction, grad))

    if torch.dot(search_direction, grad) > 0:
      print("bad direction")
      search_direction = -preconditioner(grad.clone())

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1

    print("Initiating Linesearch")
 
    stepsize, loss, x = line_search(lambda x : .5 * torch.linalg.norm(residual(x))**2 + .5 * torch.linalg.norm(D(x.clone() - m0))**2, x, search_direction, loss.item(), maxstepi, max_evals=20, print_debug=print_debug)
    grad = torch.autograd.grad(loss,x)[0]

    loss = loss.detach()
    grad = grad.detach()
    
    history.update(float(loss), float(torch.linalg.norm(grad)), parameters.targetDist(x), residual.forward_counter, residual.backward_counter)
  return x, history 


import matplotlib.pyplot as plt
 
if __name__ == "__main__":
  #old test of cg
  print("TESTING CG")
  f              = doubleRosenbrock.Sum()
  parameters     = optParams(maxiter = 1000, maxstep=4000, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history = CG(f, x0, parameters, D = lambda x : x, m0 = torch.ones(4))
  print(history.report)
  print(xstar)

  print("TESTING LBFGS")
  f              = doubleRosenbrock.Sum()
  parameters     = optParams(maxiter = 1000, maxstep=4000, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history, bfgshistory, _ = LBFGS(f, x0, parameters, D = lambda x : x, bfgsMemory = 5, m0 = torch.ones(4))
  print(history.report)
  print(xstar)

  print("TESTING SFGN")
  fs              = [doubleRosenbrock.SingleSep(i) for i in range(4)]
  parameters     = optParams(maxiter = 10, maxstep=10, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history = SFGN(fs, x0, parameters, D = lambda x : x, line_search = line_search_backtrack, m0 = torch.ones(4))
  print(history.report)
  print(xstar)

  print("TESTING GN-CG")
  f              = doubleRosenbrock.Separate()
  parameters     = optParams(maxiter = 10, maxstep=10, targetx = torch.ones(4), ltol = 1e-3)
  x0             = -torch.ones(4) 
  xstar, history = GNCG(f, x0, parameters, eta0=.08, D = lambda x : x, line_search = line_search_backtrack, m0 = torch.ones(4))
  print(history.report)
  print(xstar)


