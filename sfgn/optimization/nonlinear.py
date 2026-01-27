from typing import Callable, Any, List, Tuple
import torch
import time
from sfgn.optimization.utils import line_search_backtrack, line_search_interpolate, lbfgs_invhess, bfgsHistory, conjugate_gradient
from dataclasses import dataclass, field
from scipy.sparse.linalg import LinearOperator, eigsh

from sfgn.objectives       import doubleRosenbrock
from sfgn.objectives.utils import jacobian_with_output, parallel_jacobian, parallel_forward, parallel_backward, gnHessProd

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
  maxsim: float = 10000
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
  steplengths: List[float]  = field(default_factory=list)
 
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

  def updateStepLength(self, alpha: float):
    self.steplengths.append(alpha)

  def checkConverged(self, parameters: optParams) -> bool:
    if self.iteration > 2:
      if (abs(self.loss[-1] - self.loss[-2])/self.loss[0] <= 1e-10):
        return True
    return (self.loss[-1]/self.loss[0] <= parameters.ltol) or (self.gradnorm[-1]/self.gradnorm[0] <= parameters.gtol) or (self.iteration >= parameters.maxiter) or (self.forwards[-1] + self.backwards[-1] >= parameters.maxsim)

  def writeCSV(self, path: str):
   with open(path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Iteration", "Time", "Forwards", "Backwards", "Objective", "Gradnorm", "Model Error", "Steplength"])
    for i in range(self.iteration - 1):
      writer.writerow([i,self.times[i],self.forwards[i],self.backwards[i],self.loss[i],self.gradnorm[i],self.targetdist[i],self.steplengths[i]])
    i = self.iteration - 1
    writer.writerow([i,self.times[i],self.forwards[i],self.backwards[i],self.loss[i],self.gradnorm[i],self.targetdist[i],0])

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

  x          = x0.clone()
  x.requires_grad_()

  history    = optHist()

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
      phiprime = torch.dot(search_direction, grad)

    if torch.dot(search_direction, grad) > 0:
      with torch.no_grad():
        search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad.clone(), tol=.005, max_iter=10000)[0]
        phiprime = torch.dot(search_direction, grad)

      gamma *= 0

    search_direction.requires_grad_(False)
    gamma = search_direction.clone()

    loss_old = loss
    grad_old = grad

    maxstepi = history.gradnorm[-1] * history.gradnorm[0]**-1 * parameters.maxstep * torch.max(torch.abs(search_direction))**-1
#    maxstepi = 1

    stepsize, loss, x = line_search(lambda x : f(x) + .5 * torch.linalg.norm(D(x - m0))**2, x, search_direction, loss.item(), phiprime, maxstepi, max_evals=20, print_debug=print_debug)
    history.updateStepLength(float(stepsize))
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

  x          = x0.clone()
  x.requires_grad_()

  history    = optHist()

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
#      search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, lbfgs_invhess(grad, bfgsHist), tol=.005, max_iter=10000)[0]
#      search_direction = -lbfgs_invhess(grad, bfgsHist, init= lambda g : conjugate_gradient(lambda m : D(D(m)) + h * m, g, tol=.005, max_iter=1000)[0])
#      search_direction = -conjugate_gradient(lambda m : D(D(m)), lbfgs_invhess(grad, bfgsHist, init=lambda m : m/h), tol=.005, max_iter=10000)[0]
      search_direction = -conjugate_gradient(lambda m : D(D(m)), lbfgs_invhess(grad, bfgsHist, init= lambda g : conjugate_gradient(lambda m : D(D(m)) + h * m, g, tol=.005, max_iter=1000)[0]), tol=.005, max_iter=10000)[0]

    phiprime = torch.dot(search_direction, grad)

    if phiprime > 0:
      with torch.no_grad():
        search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad.clone(), tol=.005, max_iter=10000)[0]
        phiprime = torch.dot(search_direction, grad)
    bfgsHist   = bfgsHistory(m=bfgsMemory)
    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
    gradphi_old = gradphi

    maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1
    if torch.isinf(maxstepi):
      #print("Problem with BFGS update, resetting history")
      print(torch.max(torch.abs(grad)))
      print(torch.max(torch.abs(search_direction)))

      maxstepi = 1 
      with torch.no_grad():
        search_direction = -conjugate_gradient(lambda m : D(D(m)) + h * m, grad.clone(), tol=.005, max_iter=10000)[0]
      maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1
      phiprime = torch.dot(search_direction, grad)
    stepsize, loss, x = line_search(lambda x : f(x) + .5 * torch.linalg.norm(D(x - m0))**2, x, search_direction, loss.item(), phiprime, maxstepi, max_evals=20, print_debug=print_debug)
    history.updateStepLength(float(stepsize))
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
  D2lo: torch.Tensor | None = None,
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

  if D2lo is None:
    D2lo = torch.eye(len(h)) 

  if m0 is None:
    m0 = x0.clone().detach()

  n = len(h)
  
  def A(x):
      return D(D(x)) + h * x
  
  #def preconditioner(g):
  #  u = torch.zeros_like(g)
  #  with torch.no_grad():
  #    for _ in range(3):
  #      u = u + diaginv * (g - A(u))
  #  return u

  x          = x0.clone()
  x.requires_grad_()

  history    = optHist()

  Jac, res = parallel_jacobian(fs, x)

  loss = .5 * torch.linalg.norm(res)**2 + .5 * torch.linalg.norm(D(x.clone() - m0))**2
  grad = Jac.T @ res  

  loss = loss.detach()
  grad = grad.detach()

  history.update(float(loss) - len(fs), float(torch.linalg.norm(grad)), parameters.targetDist(x), fs[0].forward_counter, fs[0].backward_counter)


  #D2   = torch.stack([D(D(1.0*(torch.arange(n) == i))) for i in range(n)])
  #D2lo = torch.linalg.cholesky(D2)

  while not history.checkConverged(parameters):
    if print_debug:
      print(history.report)  
      print(f"Current RAM usage: {get_memory_usage():.2f} MB")

    if print_debug: 
      print(history.gradnorm[-1])
      print(torch.linalg.norm(grad))

#    def preconditioner(g):
#      u = torch.zeros_like(g)
#      for _ in range(10):
#          r = g - D(D(u)) + Jac.T@(Jac@u)  # Residual
#          u = u + omega * r
#      return u

#    search_direction, prev_solve_rnorm = conjugate_gradient(lambda v : Jac.T@(Jac@v) + D(D(v)), -grad - D(x.clone().detach() - m0), M=preconditioner, tol = cgtol, max_iter=10000)

    Jadj = torch.linalg.solve_triangular(D2lo, torch.linalg.solve_triangular(D2lo.T, Jac.T, upper = True), upper=False)

    #a = x.clone().detach() - m0 + Jadj @ res
    #M = Jac @ Jadj
    #print(f"Condition number : {torch.linalg.cond(M)}")
    #print(f"Condition number : {torch.linalg.cond(torch.eye(Jac.shape[0]) + M)}")

    #y = torch.linalg.solve(torch.eye(Jac.shape[0]) + M, Jac @ a)
    #search_direction = - ( a - Jadj @ y ) 
    dm = x.clone().detach() - m0

    M = Jac @ Jadj
    print(f"Condition number : {torch.linalg.cond(M)}")
    print(f"Condition number : {torch.linalg.cond(torch.eye(Jac.shape[0]) + M)}")

    search_direction = Jadj @ torch.linalg.solve(torch.eye(Jac.shape[0]) + M, Jac@dm - res) - dm
    
    phiprime = torch.dot(search_direction, grad)

    if phiprime > 0:
      print("bad direction")
      search_direction = -grad.clone()
      phiprime = torch.dot(search_direction, grad)
 

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = parameters.maxstep * torch.max(torch.abs(search_direction))**-1

    print("Initiating Linesearch")

    residuals, graph = [], [] 

    def calculate_loss(x):
      nonlocal residuals
      nonlocal graph
      residuals, graph = parallel_forward(fs, x) 
      return sum([.5 * r**2 for r in residuals]) + .5 * torch.linalg.norm(D(x.clone() - m0))**2

    stepsize, loss, x = line_search(calculate_loss, x, search_direction, loss.item(), phiprime, maxstepi, max_evals=20, print_debug=print_debug)
    history.updateStepLength(float(stepsize))

    Jac = torch.stack(parallel_backward(residuals, graph)) 
    res = torch.tensor(residuals)

    #stepsize, loss, x = line_search(lambda x : sum([f(x) + 1 for f in fs]) + .5 * torch.linalg.norm(D(x.clone() - m0))**2, x, search_direction, loss.item(), phiprime, maxstepi, max_evals=20, print_debug=print_debug)

    #Jac, res = parallel_jacobian(fs, x)

    loss = loss.detach()
    grad = Jac.T @ res 
    
    history.update(float(loss) - len(fs), float(torch.linalg.norm(grad)), parameters.targetDist(x), fs[0].forward_counter, fs[0].backward_counter)
  return x, history 



def GNCG( 
  residual: Callable[[torch.Tensor], torch.Tensor], 
  x0: torch.Tensor,
  parameters: optParams, 
  D: Callable[[torch.Tensor], torch.Tensor] = lambda x : x,
  D2diag: torch.Tensor | None = None,
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

  if D2diag is None:
    D2diag = torch.ones(len(h)) 

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
  
  # Choose omega conservatively to ensure convergence
  # For SPD A, convergence if 0 < ω < 2/λ_max
  
  def preconditioner(g):
      u = torch.zeros_like(g)
      for _ in range(250):
          r = g - A(u)  # Residual
          u = u + omega * r
      return u
 
  #diaginv = (D2diag + h)**-1
  #
  #def preconditioner(g):
  #  u = torch.zeros_like(g)
  #  with torch.no_grad():
  #    for _ in range(100):
  #      u = u + diaginv * (g - A(u))
  #  return u




  print(preconditioner(torch.randn(len(h))))

  x          = x0.clone()
  x.requires_grad_()

  history    = optHist()
 
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

    phiprime = torch.dot(search_direction, grad)

    if phiprime > 0:
      print("bad direction")
      search_direction = -preconditioner(grad.clone())
      phiprime = torch.dot(search_direction, grad)

    search_direction.requires_grad_(False)

    loss_old = loss
    grad_old = grad
 
    maxstepi = 1

    print("Initiating Linesearch")
 
    stepsize, loss, x = line_search(lambda x : .5 * torch.linalg.norm(residual(x))**2 + .5 * torch.linalg.norm(D(x.clone() - m0))**2, x, search_direction, loss.item(), phiprime, maxstepi, max_evals=20, print_debug=print_debug)
    history.updateStepLength(float(stepsize))

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
  parameters     = optParams(maxiter = 1000, maxstep=4000, maxsim=5000, targetx = torch.ones(4), ltol = 1e-3)
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


