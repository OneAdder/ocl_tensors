from tensors import *

print(tensor(np.array([1, 2, 3])).to_cpu())

a = tensor(np.array([1, 2, 3]))
b = tensor(np.array([1, 2, 3]))
print(a.tensor_product(b))
print(a + b)
print(a * 5)

a = tensor(np.array([[1, 2, 3], [1, 2, 3]]))
b = tensor(np.array([[1, 2, 3], [1, 2, 3]]))
print(a + b)
