import numpy as np
A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for i in range(0,-3,-1):
    A[i+3] = A[i+2]
    A[i+7] = A[i+6]
    A[i+11] = A[i+10]
    A[i+15] = A[i+14]

A[0] = 11
A[4] = 12
A[8] = 13
A[12] = 14
print(A)