from typing import Iterable, Callable, List, Tuple
import torch
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

def parallel_forward(
    misfit_modules: List[Callable[[torch.Tensor], torch.Tensor]],
    input: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Parallelize all forward computations at once"""
    
    def single_forward(i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward computation for a single module"""
        with torch.no_grad():
            m_local = input.detach().clone()
            m_local.requires_grad_()
        
        # Forward computation
        misfit = misfit_modules[i](m_local)
        residual = (2 * (misfit + 1)) ** 0.5
        
        return residual, m_local
    
    # Execute ALL forward computations in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(single_forward, range(len(misfit_modules))))
    
    # Separate residuals and computation graphs
    residuals = [r for r, _ in results]
    computation_graphs = [g for _, g in results]
    
    return residuals, computation_graphs

def parallel_backward(
    residuals: List[torch.Tensor],
    computation_graphs: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallelize all backward computations at once"""
    
    def single_backward(i: int) -> torch.Tensor:
        """Backward computation for a single residual"""
        # Perform backward pass
        residuals[i].backward()
        gradient = computation_graphs[i].grad.clone()
        
        # Clean up gradient to avoid memory issues
        if computation_graphs[i].grad is not None:
            computation_graphs[i].grad.zero_()
        
        return gradient
    
    # Execute ALL backward computations in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        gradients = list(executor.map(single_backward, range(len(residuals))))
    
    return gradients
 
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

