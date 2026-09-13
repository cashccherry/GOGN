from typing import Callable, Optional, List, Tuple

import torch
from sfgn.regularization.regularizer import Regularizer

class TVRegularizer(Regularizer):
  def __init__(
   self, 
   m0: torch.Tensor,
   Dx: Callable[[torch.Tensor], torch.Tensor], 
   Dy: Callable[[torch.Tensor], torch.Tensor], 
   Dxt: Callable[[torch.Tensor], torch.Tensor], 
   Dyt: Callable[[torch.Tensor], torch.Tensor], 
   L: Callable[[torch.Tensor], torch.Tensor], 
   ε: float,   
   λ: float,   
   μ: float,
   ν: float,
   h: float,    
  ):
    super().__init__()
    self.m0   = m0
    self.Dx   = Dx
    self.Dy   = Dy
    self.Dxt  = Dxt
    self.Dyt  = Dyt
    self.L    = L 
    self.ε    = ε
    self.λ    = λ
    self.μ    = μ
    self.ν    = ν 
    self.h    = h

  def forward(self, m: torch.Tensor):
    TVnorm = torch.sum((self.Dx(m)**2 + self.Dy(m)**2 + self.ε**2)**.5) - torch.sum(1 + 0*self.m0)*self.ε
    return self.h*(self.λ*TVnorm + self.μ*torch.linalg.norm(m - self.m0)**2/2 + self.ν*torch.linalg.norm(self.L(m - self.m0))**2/2)

  def grad(self, m: torch.Tensor):
    gx = self.Dx(m)
    gy = self.Dy(m)

    w = (gx**2 + gy**2 + self.ε**2)**.5

    TVgrad = self.Dxt(gx/w) + self.Dyt(gy/w)
 
    return self.h*(self.λ * TVgrad + self.μ*(m - self.m0) + self.ν*self.L(self.L(m - self.m0)))

  def hess_vec(self, m, v):
    gx = self.Dx(m)
    gy = self.Dy(m)

    vx = self.Dx(v)
    vy = self.Dy(v)

    w = torch.sqrt(gx**2 + gy**2 + self.ε**2)

    inner = gx * vx + gy * vy

    qx = vx / w - gx * inner / w**3
    qy = vy / w - gy * inner / w**3

    TVhv = self.Dxt(qx) + self.Dyt(qy)

    return self.h*(self.λ * TVhv + self.μ * v + self.ν * self.L(self.L(v)))

#  def hess_solve(self, m: torch.Tensor, b: torch.Tensor):
#    if self.D2lo is None:
#      print("Using CG")
#      return super().hess_solve(m, b) 
#    if len(b.shape) == 1:
#      return self.hess_solve(m, b.unsqueeze(1)).flatten()
#    return torch.linalg.solve_triangular(self.D2lo, torch.linalg.solve_triangular(self.D2lo.T, b, upper = True), upper=False).detach()
      



