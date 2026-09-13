from typing import Callable, Optional, List, Tuple

import torch
from sfgn.regularization.regularizer import Regularizer

class TikhonovRegularizer(Regularizer):
  def __init__(
   self, 
   m0: torch.Tensor,
   Dx: Callable[[torch.Tensor], torch.Tensor], 
   Dy: Callable[[torch.Tensor], torch.Tensor], 
   Dxt: Callable[[torch.Tensor], torch.Tensor], 
   Dyt: Callable[[torch.Tensor], torch.Tensor], 
   L: Callable[[torch.Tensor], torch.Tensor], 
   λ0: float,   
   λ1: float,   
   λ2: float,   
   h: float,    
  ):
    super().__init__()
    self.m0   = m0
    self.Dx   = Dx
    self.Dy   = Dy
    self.Dxt  = Dxt
    self.Dyt  = Dyt
    self.L    = L 
    self.λ0   = λ0
    self.λ1   = λ1
    self.λ2   = λ2
    self.h    = h

  def forward(self, m: torch.Tensor):
    return self.h*(
      self.λ0*torch.linalg.norm(m - self.m0)**2/2 + 
      self.λ1*torch.linalg.norm(self.Dx(m - self.m0))**2/2 + 
      self.λ1*torch.linalg.norm(self.Dy(m - self.m0))**2/2 + 
      self.λ2*torch.linalg.norm(self.L(m - self.m0))**2/2
    )

  def grad(self, m: torch.Tensor):
    return self.h*(
      self.λ0*(m - self.m0) + 
      self.λ1*self.Dxt(self.Dx(m - self.m0)) + 
      self.λ1*self.Dyt(self.Dy(m - self.m0)) + 
      self.λ2*self.L(self.L(m - self.m0))
    )


  def hess_vec(self, m, v):
    return self.h*(
      self.λ0*v + 
      self.λ1*self.Dxt(self.Dx(v)) + 
      self.λ1*self.Dyt(self.Dy(v)) + 
      self.λ2*self.L(self.L(v))
    )



