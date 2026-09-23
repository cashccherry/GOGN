from typing import Callable, Any, List, Tuple
import torch
from gogn.optimization.utils import conjugate_gradient

class Regularizer(torch.nn.Module):
  def forward(self, m: torch.Tensor):
    return 0. * torch.sum(m) 

  def grad(self, m: torch.Tensor) -> torch.Tensor:
    m = m.detach().requires_grad_(True)

    g, = torch.autograd.grad(
        self(m),
        m,
        create_graph=False,
    )
    return g

  def hess_vec(self, m: torch.Tensor, v: torch.Tensor):
    m = m.detach().requires_grad_(True)

    g = torch.autograd.grad(self(m), m, create_graph=True)[0]

    Hv = torch.autograd.grad(
        g,
        m,
        grad_outputs=v,
        retain_graph=False
    )[0]

    return Hv   

  def hess_solve_grad(self, m: torch.Tensor):
    """Return (∇²R(m))⁻¹∇R(m) for the general Woodbury direction."""
    with torch.enable_grad():
      point = m.detach()
      gradient = self.grad(point).detach()
      return self.hess_solve(point, gradient).detach()

  def hess_solve(self, m: torch.Tensor, b: torch.Tensor):
    Hv = lambda v: self.hess_vec(m, v)

    if b.ndim == 1:
      return conjugate_gradient(Hv, b, tol=0.005, max_iter=100)[0]

    elif b.ndim == 2:
      solutions = []
      for i in range(b.shape[1]):
        x_i = conjugate_gradient(
          Hv,
          b[:, i],
          tol=0.005,
          max_iter=100
        )[0]
        solutions.append(x_i)

      return torch.stack(solutions, dim=1)

    else:
      raise ValueError(f"Expected vector or matrix b, got shape {b.shape}")

