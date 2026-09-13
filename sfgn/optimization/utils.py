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
    if (loss_val < losses[0] or i == max_evals - 1) and len(steps) > 1:
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
    if (len(steps) >= 2 and len(steps) < 6):
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
    return step, loss_new, x_new


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

    if torch.dot(Ap, p) < -1e-7 * torch.dot(p, p):
      print(f"Negative Curvature Detected in Conjugate Gradient :  {torch.dot(Ap, p) / torch.dot(p,p)}")
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

    alpha = rsq / torch.dot(p, Ap)
    x    += alpha * p
    r    += alpha * Ap 

    z = M(r)

    beta = torch.dot(r,z) / rsq
    p = -z + beta * p 
  print("CG DONE") 
  return x, rnorm


import torch

class LimitedMemoryStructuredSecantTorch:
    """
    Limited-memory implementation of the Dennis-Gay-Welsch structured secant update
    (eq. 10.43) using PyTorch tensors. Suitable for large-scale nonlinear least squares
    on CPU or GPU.

    - Approximates the second-order term S ≈ sum r_j ∇² r_j implicitly.
    - Supports fast matrix-vector products: S @ v in O(m n) time.
    - Enforces the secant condition exactly: S_new @ s ≈ y_hash (up to numerical precision).
    - Initial approximation: S_0 = gamma * I.
    """
    def __init__(
        self,
        n: int,
        m: int = 10,
        gamma: float = 1.0,
        update_gamma: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float64,  # or torch.float32 for faster but less precise
    ):
        """
        Args:
            n: Dimension of the variables (length of x, s, y, etc.).
            m: Maximum number of stored updates (memory parameter, typically 5–20).
            gamma: Initial scaling for S_0 = gamma * I.
            update_gamma: If True, set gamma ← (y^T s) / ||y||² at each step.
            device: 'cuda' or 'cpu'.
            dtype: torch.float64 (recommended for quasi-Newton stability) or torch.float32.
        """
        self.n = n
        self.m_max = m
        self.gamma = gamma
        self.update_gamma = update_gamma
        self.device = torch.device(device)
        self.dtype = dtype

        # History lists: oldest first
        self.ys = []       # y vectors
        self.zs = []       # z = y_hash - S_prev @ s
        self.denoms = []   # y^T s
        self.sigmas = []   # z^T s

    def _matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute current S @ v using the stored updates."""
        v = v.to(dtype=self.dtype)

        q = self.gamma * v.clone()
        for y, z, denom, sigma in zip(self.ys, self.zs, self.denoms, self.sigmas):
            if torch.abs(denom) < 1e-14:
                continue  # skip numerically unstable update

            yTq = torch.dot(y, q)
            zTq = torch.dot(z, q)

            term1 = (zTq / denom) * y
            term2 = (yTq / denom) * z
            term3 = (sigma * yTq / (denom ** 2)) * y

            q = q + term1 + term2 - term3

        return q

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Public method: compute S @ v."""
        v = v.to(dtype=self.dtype)

        if v.device != self.device:
            v = v.to(self.device, dtype=self.dtype)
        return self._matvec(v)

    def update(
        self,
        s: torch.Tensor,
        y: torch.Tensor,
        y_hash: torch.Tensor,
    ):
        """
        Perform one update using formula (10.43).

        Args:
            s: step = x_{k+1} - x_k          (shape (n,))
            y: J_{k+1}^T r_{k+1} - J_k^T r_k (shape (n,))
            y_hash: J_{k+1}^T r_{k+1} - J_k^T r_{k+1} (shape (n,))
        """
        # Move to correct device/dtype if needed
        s = s.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)
        y_hash = y_hash.to(self.device, dtype=self.dtype)

        # Compute z = y_hash - S_k @ s
        S_s = self._matvec(s)
        z = y_hash - S_s

        denom = torch.dot(y, s)
        if torch.abs(denom) < 1e-14:
            print("Warning: y^T s too small; update skipped.")
            return

        sigma = torch.dot(z, s)

        # Optional: update scaling (common quasi-Newton heuristic)
        if self.update_gamma:
            y_norm_sq = torch.dot(y, y)
            if y_norm_sq > 1e-14:
                self.gamma = denom / y_norm_sq  # or denom / y_norm_sq (BFGS-like scaling)

        # Store the new update
        self.ys.append(y.clone())
        self.zs.append(z.clone())
        self.denoms.append(denom)
        self.sigmas.append(sigma)

        # Enforce limited memory
        if len(self.ys) > self.m_max:
            self.ys.pop(0)
            self.zs.pop(0)
            self.denoms.pop(0)
            self.sigmas.pop(0)

    def reset(self):
        """Clear history and reset to initial approximation."""
        self.ys.clear()
        self.zs.clear()
        self.denoms.clear()
        self.sigmas.clear()
        # gamma can be kept or reset to initial value if desired



if __name__ == "__main__":
  print(20*'~')
  print("TESTING LINESEARCH AND QUADRATIC INTERPOLATION")
  print(20*'~')

  print("QUADRATIC TEST FUNCTION, LINESEARCH WILL SUCCEED")

  f = lambda x : (x - 2)**2 + 1
  x = [-1, 15, 10]
  y = [f(xi) for xi in x]

  interp_result = (quadratic_interpmin(x,y))
  assert abs(interp_result - 2) <= 1e-6
  #print(torch.tensor(line_search_interpolate(f, torch.tensor([0]), torch.tensor([1]), f(0), 5, 10)))
  assert torch.linalg.norm(torch.tensor(line_search_interpolate(f, torch.tensor([0]), torch.tensor([1]), f(0), 5, 10))[:-1] - torch.tensor([2,1])) <= 1e-5

  print("LINEAR TEST FUNCTION, WILL FAIL LINESEARCH")

  f = lambda x : x
  k = [-1, 15, 10]
  y = [f(xi) for xi in x]

  assert abs(quadratic_interpmin(x,y)) > 1e10
  assert line_search_interpolate(f, torch.tensor([0]), torch.tensor([1]), f(0), 5, 10)[0] > 0
  assert line_search_interpolate(f, torch.tensor([0]), torch.tensor([1]), f(0), 5, 10)[1] > f(0)

  print(50*"-")
  """Test basic L-BFGS inverse Hessian operations"""
  print("\n" + "="*50)
  print("TESTING LBFGS_INVHESS")
  
  # Test 1: Empty history should return input unchanged
  history = bfgsHistory(m=None)
  p = torch.ones(15)
  result = lbfgs_invhess(p, history)
  assert torch.equal(result, p), "Empty history should return input unchanged"
  print("✓ Empty history test passed")
  
  # Test 2: Single update
  history.update(p, 10*p)
  result1 = lbfgs_invhess(p, history)
  print(f"Single update result: {result1}")
  
  # Test 3: Multiple updates
  history.update(p, 2*p)
  result2 = lbfgs_invhess(p, history)
  print(f"Double update result: {result2}")
  
  # Test 4: Verify results are different
  assert not torch.equal(result1, result2), "Different histories should produce different results"
  print("✓ Multiple update test passed")

  # Test linearity/scaling
  history = bfgsHistory(m=5)
  v = torch.randn(10, requires_grad=False)
  
  # Test scaling property: H(αv) should be approximately αH(v) for small α
  alpha = 2.0
  result_scaled = lbfgs_invhess(alpha * v, history)
  result_then_scaled = alpha * lbfgs_invhess(v, history)
  
  # They might not be exactly equal due to history updates, but should be close
  # for empty history or consistent updates
  history_empty = bfgsHistory(m=None)
  result1 = lbfgs_invhess(alpha * v, history_empty)
  result2 = alpha * lbfgs_invhess(v, history_empty)
  assert torch.allclose(result1, result2), "Should satisfy scaling for empty history"
  print("✓ Scaling property test passed")

  torch.manual_seed(42)  # For reproducible tests
  
  history = bfgsHistory(m=3)
  n = 8  # Smaller dimension for easier testing
  
  # Generate some BFGS updates
  for i in range(3):
      s = torch.randn(n)  # Step
      y = torch.randn(n)  # Gradient difference  
      history.update(s, y)
  
  # Test secant condition: H*y should be approximately s
  # This is the fundamental BFGS condition: H_{k+1} * y_k = s_k
  s, y = history.S[-1], history.Y[-1]
  H_y = lbfgs_invhess(y, history)
  
  # They should be approximately equal (within numerical tolerance)
  tolerance = 1e-6
  assert torch.allclose(H_y, s, atol=tolerance), f"Secant condition not satisfied: {H_y} vs {s}"
  print("✓ Secant condition test passed")

  print("Testing Conjugate Gradient on Interesting Problems")

  Lapmem = bfgsHistory()
   
  # Test 1: Sparse banded matrix (simulating finite difference Laplacian)
  print("\n1. Sparse banded matrix (1D Laplacian):")
  n = 40
  # A = tridiagonal matrix with 2 on diagonal, -1 on off-diagonals (discrete 1D Laplacian)
  def laplacian_1d(x):
      result = 2 * x.clone()
      result[1:] -= x[:-1]  # -1 * lower diagonal
      result[:-1] -= x[1:]  # -1 * upper diagonal
      return result
  
  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(laplacian_1d, b1, memory=Lapmem, print_debug=True)
  print(f"Final residual norm: {rnorm1:.2e}")

  print("\n1. Sparse banded matrix with memory from previous solve (1D Laplacian):")

  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(laplacian_1d, b1, M=lambda m : lbfgs_invhess(m, Lapmem), print_debug=True)
  print(f"Final residual norm: {rnorm1:.2e}")

  Lapmem = bfgsHistory(5)

  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(laplacian_1d, b1, memory=Lapmem, print_debug=False)

  print("\n1. Sparse banded matrix with limited memory from previous solve (1D Laplacian):")

  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(laplacian_1d, b1, M=lambda m : lbfgs_invhess(m, Lapmem), print_debug=True)
  print(f"Final residual norm: {rnorm1:.2e}")


  print("\n1. Test of regularizing the preconditioner (1D Laplacian):")

  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(lambda m : laplacian_1d(m) + m, b1, print_debug=True)
  print(f"Final residual norm no prec: {rnorm1:.2e}")


  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(lambda m : laplacian_1d(m) + m, b1, M=lambda m : lbfgs_invhess(m, Lapmem), print_debug=True)
  print(f"Final residual norm prec: {rnorm1:.2e}")


  b1 = torch.eye(n)[1]  # Standard basis vector
  x1, rnorm1 = conjugate_gradient(lambda m : laplacian_1d(m) + m, b1, M=lambda m : lbfgs_invhess(m, Lapmem, tikhReg=1), print_debug=True)
  print(f"Final residual norm reg prec: {rnorm1:.2e}")




  
  # Test 2: Random positive definite matrix
  print("\n2. Random positive definite matrix:")
  torch.manual_seed(42)
  M = torch.randn(n, n)
  A_mat = M @ M.T + torch.eye(n)  # Make it positive definite
  def random_pd(x):
      return A_mat @ x
  
  b2 = torch.randn(n)
  x2, rnorm2 = conjugate_gradient(random_pd, b2, print_debug=True)
  print(f"Final residual norm: {rnorm2:.2e}")
  
  # Test 3: Diagonal matrix with varying condition number
  print("\n3. Ill-conditioned diagonal matrix:")
  # Create diagonal matrix with eigenvalues from 1 to 1000 (condition number ~1000)
  diag = torch.linspace(1, 1000, n)
  def ill_conditioned(x):
      return diag * x
  
  b3 = torch.ones(n)  # All ones vector
  x3, rnorm3 = conjugate_gradient(ill_conditioned, b3, print_debug=True)
  print(f"Final residual norm: {rnorm3:.2e}")
  
  # Test 4: Check solution accuracy for one case
  print("\n4. Verification for diagonal system:")
  # Solve Dx = b where D is known diagonal
  exact_solution = b3 / diag
  print(f"Max error in solution: {torch.max(torch.abs(x3 - exact_solution)):.2e}")
