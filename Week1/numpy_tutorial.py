import numpy as np

## Arrays
a = np.array([1, 2, 3]) # Rank 1 array
print type(a)
print a.shape
print a[0], a[1], a[2]
a[0] = 5
print a

b = np.array([[1,2,3],[4,5,6]]) # Rank 2 array
print b.shape
print b[0, 0], b[0, 1], b[1, 0]