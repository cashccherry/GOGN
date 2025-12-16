import torch
import torch.nn as nn

class Separate(nn.Module):
  def __init__(self):
    super().__init__()
    self.forward_counter  = 0
    self.backward_counter = 0

  def forward(self, x):
    self.forward_counter += 1

    # Use PyTorch operations
    part1 = 1 - x[0]
    part2 = 100 * (x[1] - x[0]) 
    part3 = 1 - x[2]
    part4 = 100 * (x[3] - x[2])
    
    output = torch.stack([part1, part2, part3, part4])**2

    if x.requires_grad:
      output.register_hook(self._backward_hook)
    return output

  def _backward_hook(self, grad):
    self.backward_counter += 1
    return grad

class SingleSep(Separate):
  def __init__(self, index: int):
    super().__init__()
    self.index = index

  def forward(self, x):
    outputs = super().forward(x)
    return outputs[self.index]

class Sum(nn.Module):
  def __init__(self):
    super().__init__()
    self.separate = Separate()

  def forward(self, x):
    outputs = self.separate(x)
    return torch.sum(outputs)
    
  @property
  def forward_counter(self):
    return self.separate.forward_counter

  @property
  def backward_counter(self):
    return self.separate.backward_counter

class CombineRosens(nn.Module):
  def __init__(self):
    super().__init__()
    self.separate = Separate()

  def forward(self, x):
    outputs = self.separate(x)
    return torch.stack([outputs[0] + outputs[1], outputs[2] + outputs[3]])

  def _backward_hook(self, grad, name):
    self.separate.backward_counter += 1
    return grad
    
  @property
  def forward_counter(self):
    return self.separate.forward_counter

  @property
  def backward_counter(self):
    return self.separate.backward_counter
    
class CombineLike(nn.Module):
  def __init__(self):
    super().__init__()
    self.separate = Separate()

  def forward(self, x):
    outputs = self.separate(x)
    return torch.stack([outputs[0] + outputs[2], outputs[1] + outputs[3]])
    
  @property
  def forward_counter(self):
    return self.separate.forward_counter

  @property
  def backward_counter(self):
    return self.separate.backward_counter
    
  
if __name__ == "__main__":
  print(20*"-")
  print("UNIT TESTS OF DOUBLE ROSENBROCK FUNCTION")
  print(20*"-")

  print("TESTING SEPARATE OBJECTIVES")
  xmin  = torch.ones(4)
  xmin.requires_grad_()
  separate = Separate()
  assert separate.forward_counter == 0 # We have computed no forward passes
  assert torch.equal(separate(xmin), torch.zeros(4)) # Assert minimizer has value 0
  assert separate.forward_counter == 1 # We have computed one forward pass
  assert separate.backward_counter  == 0 # We have computed no backward pass yet
  assert torch.equal(torch.autograd.functional.jacobian(separate, xmin), torch.zeros(4,4)) # Assert correct Jacobian at minimizer
  assert separate.forward_counter == 2 # We have computed two forward passes
  assert separate.backward_counter == 4 # We have computed four backward passes
  del separate
  print("SEPARATE OBJECTIVES PASSED")

  print(20*"-")

  print("TESTING SINGLE SEPARATE OBJECTIVES")
  xmin  = torch.ones(4)
  xmin.requires_grad_()
  separate = SingleSep(2)
  assert separate.forward_counter == 0 # We have computed no forward passes
  assert torch.equal(separate(xmin), torch.tensor(0.)) # Assert minimizer has value 0
  assert separate.forward_counter == 1 # We have computed one forward pass
  assert separate.backward_counter  == 0 # We have computed no backward pass yet
  assert torch.equal(torch.autograd.functional.jacobian(separate, xmin), torch.zeros(4)) # Assert correct Jacobian at minimizer
  assert separate.forward_counter == 2 # We have computed two forward passes
  assert separate.backward_counter == 1 # We have computed four backward passes
  del separate
  print("SEPARATE OBJECTIVES PASSED")

  print(20*"-")

  print("TESTING SUM OF OBJECTIVES")
  xmin  = torch.ones(4)
  xmin.requires_grad_()
  sumobj = Sum()
  assert sumobj.forward_counter == 0
  outsum = sumobj(xmin)
  assert sumobj.forward_counter == 1
  outsum.backward()
  assert torch.equal(xmin.grad, torch.zeros(4)) 
  assert sumobj.forward_counter == 1
  assert sumobj.forward_counter == 1
  print("SUM OF OBJECTIVES PASSED")

  print(20*"-")

  print("TESTING COMBINING ROSENBROCKS") 
  xmin  = torch.ones(4)
  xmin.requires_grad_()
  combine = CombineRosens()
  assert combine.forward_counter == 0 # We have computed no forward passes
  outcomb = combine(xmin)
  assert torch.equal(outcomb, torch.zeros(2)) # Assert minimizer has value 0
  assert combine.forward_counter == 1 # We have computed one forward pass
  assert combine.backward_counter  == 0 # We have computed no backward pass yet
  assert torch.equal(torch.autograd.functional.jacobian(combine, xmin), torch.zeros(2,4)) # Assert correct Jacobian at minimizer
  assert combine.forward_counter == 2 # We have computed two forward passes
  assert combine.backward_counter == 2 # We have computed four backward passes
  assert torch.equal(torch.func.jvp(combine, (xmin,), (torch.ones(4),))[1], torch.zeros(2))
  assert combine.forward_counter == 3 # We have computed two forward passes
  assert combine.backward_counter == 2 # We have computed four backward passes
  assert torch.equal(torch.func.vjp(combine, xmin)[1](torch.ones(2))[0], torch.zeros(4))
  assert combine.forward_counter == 4 # We have computed two forward passes
  assert combine.backward_counter == 3 # We have computed four backward passes
  del combine
  print("COMBINING ROSENBROCKS PASSED")

  print(20*"-")

  print("TESTING COMBINING LIKE TERMS") 
  xmin  = torch.ones(4)
  xmin.requires_grad_()
  combine = CombineLike()
  assert combine.forward_counter == 0 # We have computed no forward passes
  outcomb = combine(xmin)
  assert torch.equal(outcomb, torch.zeros(2)) # Assert minimizer has value 0
  assert combine.forward_counter == 1 # We have computed one forward pass
  assert combine.backward_counter  == 0 # We have computed no backward pass yet
  assert torch.equal(torch.autograd.functional.jacobian(combine, xmin), torch.zeros(2,4)) # Assert correct Jacobian at minimizer
  assert combine.forward_counter == 2 # We have computed two forward passes
  assert combine.backward_counter == 2 # We have computed four backward passes
  assert torch.equal(torch.func.jvp(combine, (xmin,), (torch.ones(4),))[1], torch.zeros(2))
  assert combine.forward_counter == 3 # We have computed two forward passes
  assert combine.backward_counter == 2 # We have computed four backward passes
  assert torch.equal(torch.func.vjp(combine, xmin)[1](torch.ones(2))[0], torch.zeros(4))
  assert combine.forward_counter == 4 # We have computed two forward passes
  assert combine.backward_counter == 3 # We have computed four backward passes
  del combine
  print("COMBINING LIKE TERMS PASSED")

  print(20*"-")

  print("ALL DOUBLE ROSENBROCK TESTS PASSED")

  print(20*"-")

