from tensors import *

a = tensor(np.array([1, 2, 3]))
b = tensor(np.array([1, 2, 3]))
print(a.tensor_product(b))

print(tensor(np.array([1, 2, 3])).to_cpu())
