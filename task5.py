from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Функция для приведения матрицы к верхнетреугольному виду методом Гаусса
def gaussian_elimination(A, b):
    n = len(b)
    
    for pivot_row in range(n):
        pivot_val = A[pivot_row][pivot_row]
        
        for row in range(pivot_row + 1, n):
            factor = A[row][pivot_row] / pivot_val
            A[row] -= factor * A[pivot_row]
            b[row] -= factor * b[pivot_row]

    return A, b

# Генерация случайной матрицы и вектора
def generate_random_system(n):
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    
    return A, b

sizes = [100, 200, 500]

for n in sizes:
    A, b = generate_random_system(n)
    
    local_n = n // size
    local_start = rank * local_n
    local_end = local_start + local_n

    start_time = time.time()

    local_A, local_b = A[local_start:local_end], b[local_start:local_end]
    local_A, local_b = gaussian_elimination(local_A, local_b)

    results = comm.gather((local_A, local_b), root=0)

    if rank == 0:
        combined_A = np.concatenate([result[0] for result in results])
        combined_b = np.concatenate([result[1] for result in results])

        end_time = time.time()
        print(f"Size {n}x{n}: Time - {end_time - start_time} sec. Result: {results}")
