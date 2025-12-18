from typing import Iterable, Callable, List, Tuple
import torch
from sfgn.objectives.doubleRosenbrock import Separate
import concurrent.futures


def gnHessProd(residual: Callable[[torch.Tensor], torch.Tensor], input: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Compute product with the Gauss-Newton Hessian J.T J of the residual"""
  input = input.detach()

  with torch.enable_grad():
    input.requires_grad_()
    res   = residual(input)
    _, Jv = torch.autograd.functional.jvp(
      residual, input, v, create_graph = False
    )
    JtJv = torch.autograd.grad(
      res, input, grad_outputs=Jv.detach(), retain_graph=False, create_graph = False
    )[0]
  return JtJv
 
def parallel_jacobian(misfit_modules: List[Callable[[torch.Tensor], torch.Tensor]], input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute misfits and their gradients efficiently, returning the Jacobian"""
  def compute_residual_grad(i):
       # Need to create a fresh computation graph for each thread
       with torch.no_grad():
           m_local = input.detach().clone()
           m_local.requires_grad_()
       res = (2 * (misfit_modules[i](m_local) + 1))**.5
       res.backward()
       return res, m_local.grad.clone()
   
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results = list(executor.map(compute_residual_grad, range(len(misfit_modules))))
  
  reslist, Jaclist = zip(*results)  # Unpack results
  reslist = list(reslist)
  Jaclist = list(Jaclist)
  
  return torch.stack(Jaclist), torch.Tensor(reslist) 

def jacobian_with_output(func, inputs):
    """Compute Jacobian and capture output during computation"""
    output_container = [None]
    
    def wrapped_func(*args):
        result = func(*args)
        output_container[0] = result.detach().clone()
        return result
    
    jacobian = torch.autograd.functional.jacobian(wrapped_func, inputs)
    return jacobian, output_container[0]

if __name__ == "__main__":
  print(20*"-")
  print("TESTING JACOBIAN WITH OUTPUT")
  print(20*"-")

  print("TESTING SEPARATE OBJECTIVES DOUBLE ROSENBROCK")
  xmin  = torch.ones(4)
  xmin.requires_grad_()
  separate = Separate()
  assert separate.forward_counter == 0 # We have computed no forward passes
  print(jacobian_with_output(separate, xmin))
  print(separate.forward_counter, separate.backward_counter)
  del separate

  xmin  = torch.randn(4)
  xmin.requires_grad_()
  separate = Separate()
  assert separate.forward_counter == 0 # We have computed no forward passes
  print(jacobian_with_output(separate, xmin))
  print(separate.forward_counter, separate.backward_counter)
  del separate
  print("PASSED")

  print("Testing FWI")

  from sfgn.objectives.FWI import *
  from sfgn.optimization.utils import line_search_interpolate, line_search_backtrack
  from sfgn.optimization.nonlinear import *
  
  src      = torch.tensor(np.load(os.path.join(CWD_file, "FWI/bad/srcs.npy")))
  rec      = torch.tensor(np.load(os.path.join(CWD_file, "FWI/bad/recs.npy")))
  
  invPars  = InversionPars(sigmak = 13e3)
  #invPars  = InversionPars(sigmak = 9e3)
  #print(invPars.eps)
  #quit()
  
  
  simPars   = SimulationPars(src = src, rec = rec)
  m_true    = simPars.load_image_as_velocity_field()
  F, d_obs, Fs  = simPars.get_F_data(m_true)
  weights   = simPars.get_geom_weights(d_obs, invPars.kernel)
  
  
  m0        = 0*m_true.clone()
  m0.requires_grad_()
  
  f       = waveformMisfit(F, d_obs, weights)
  fs      = [waveformMisfit(Fs[i], d_obs[[i]], weights[[i]]) for i in range(len(Fs))]
 
  print([p.shape for p in parallel_jacobian(fs, m0)]) 


  print("Testing Gauss Newton Hessian Product")
  residual = fullResidual(F, d_obs, weights)

  m0        = 0*m_true.clone()
  m0.requires_grad_()

  R = residual(m0)
  
  prod_vec = torch.ones_like(m_true)#1.0*(torch.arange(len(m_true)) == 1000)
  prod_vec = 1.0*(torch.arange(len(m_true)) == 18278)
 
  product  = gnHessProd(residual, m0, prod_vec)
  print(product)
  #import matplotlib.pyplot as plt
  #plt.figure()
  #simPars.plot_grad(product, plt.gca())
  #plt.show() 


