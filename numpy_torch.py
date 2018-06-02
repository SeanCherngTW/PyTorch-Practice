import torch
import numpy as np

# numpy to torch tensor
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
torch2np = torch_data.numpy()
print(
    'numpy\n', np_data,
    '\ntorch\n', torch_data,
    '\ntorch2np\n', torch2np
)

# list to torch tensor
list_data = [1, -2, 3, -4]
torch_data = torch.FloatTensor(list_data)  # float-32
print(
    'list\n', list_data,
    '\ntorch\n', torch_data,
)

# Matrix multiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # float-32
print(
    'mm\n', torch.mm(tensor, tensor),
    '\navg\n', torch.mean(tensor)
)
