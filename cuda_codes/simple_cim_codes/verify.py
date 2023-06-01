import numpy as np

M = int(input("Input M: "))
K = int(input("Input K: "))
N = int(input("Input N: "))

A = [[0 for _ in range(K)] for __ in range(M)]
B = [[0 for _ in range(N)] for __ in range(K)]
C = [[0 for _ in range(N)] for __ in range(M)]

A = np.array(A)
B = np.array(B)
C = np.array(C)

A = A.astype(np.float16)
B = B.astype(np.float16)
C = C.astype(np.float16)

print("Matrix A:")
for i in range(M):
    for j in range(K):
        A[i][j] = np.float16(((i * K + j) % 13) / 13)
        print(A[i][j], end=' ')
    print()

print("Matrix B:")
for i in range(N):
    for j in range(K):
        B[j][i] = np.float16(((i * K + j) % 14) / 14)
for i in range(K):
    for j in range(N):
        print(B[i][j], end=' ')
    print()

# C = np.matmul(A, B)

print("Matrix C:")
for i in range(M):
    for j in range(N):
        C[i][j] = 0
        for k in range(K):
            C[i][j] += np.float16(A[i][k] * B[k][j])
        print(C[i][j], end=' ')
    print()

        