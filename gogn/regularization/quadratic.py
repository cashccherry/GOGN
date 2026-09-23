from typing import Callable, Optional, List, Tuple

import torch
from gogn.regularization.regularizer import Regularizer

class QuadraticRegularizer(Regularizer):
  def __init__(
   self, 
   m0: torch.Tensor,
   D: Callable[[torch.Tensor], torch.Tensor], 
   Dt: Callable[[torch.Tensor], torch.Tensor] | None = None, 
   D2lo: torch.Tensor | None = None
  ):
    super().__init__()
    self.D    = D
    self.m0   = m0
    self.D2lo = D2lo

    if Dt is None:
      self.Dt = D
    else:
      self.Dt = Dt

  def forward(self, m: torch.Tensor):
    return torch.linalg.norm(self.D(m - self.m0))**2 / 2

  def grad(self, m: torch.Tensor):
    return self.Dt(self.D(m - self.m0))

  def hess_vec(self, m: torch.Tensor, v: torch.Tensor):
    return self.Dt(self.D(v))

  def hess_solve_grad(self, m: torch.Tensor):
    """For this centered quadratic, B⁻¹∇R = m−m0; no operator/solve needed."""
    return (m - self.m0).detach()

  def hess_solve(self, m: torch.Tensor, b: torch.Tensor):
    """Solve (DᵀD) x = b; D2lo is its lower Cholesky factor L."""
    if self.D2lo is None:
      print("Using CG")
      return super().hess_solve(m, b) 
    if len(b.shape) == 1:
      return self.hess_solve(m, b.unsqueeze(1)).flatten()
    # DᵀD = L Lᵀ: first solve L y = b, then Lᵀ x = y.
    y = torch.linalg.solve_triangular(self.D2lo, b, upper=False)
    return torch.linalg.solve_triangular(self.D2lo.T, y, upper=True).detach()
      



