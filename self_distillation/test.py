from print_color import *
import torch

import torch

# Example tensors
A = torch.randn(4, 4, 1)  # Tensor A with shape [300, 110, 3]
binary_mask = torch.randint(0, 2, (4, 4)).bool()  # Binary mask with shape [300, 110]

# Setting the values in A corresponding to the True values in binary_mask to 0
A[binary_mask] = 0

print(binary_mask)
print(A)
