from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sizes = [100, 200, 500]

for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    C = np.zeros((n, n))

    local_n = n // size
    local_start = rank * local_n
    local_end = local_start + local_n

    start_time = time.time()

    for i in range(local_start, local_end):
        for j in range(n):
            C[i] += A[i] * B[:, j]

    end_time = time.time()

    result = comm.gather(C[local_start:local_end], root=0)

    if rank == 0:
        final_result = np.concatenate(result)
        print(f"Size of matrix {n}x{n}: Time - {end_time - start_time} sec. Result {C}")
