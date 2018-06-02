"""
output:
tensor([[ 1.,  2.],
        [ 3.,  4.]])
tensor([[  1.,   4.],
        [  9.,  16.]])
tensor(7.5000)
tensor([[ 0.5000,  1.0000],
        [ 1.5000,  2.0000]])
tensor([[ 1.,  2.],
        [ 3.,  4.]])
[[1. 2.]
 [3. 4.]]
"""
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)  # True: compute gradient
v_out = torch.mean(variable * variable)
v_out.backward()

print(variable)
print(variable * variable)  # matrix multiplication
print(torch.mean(variable * variable))

print(variable.grad)
print(variable.data)  # variable to tensor
print(variable.data.numpy())  # variable to tensor to numpy
