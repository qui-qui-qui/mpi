from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sizes = [100, 1000]

for n in sizes:
    m = n

    A = np.zeros((n, m), dtype=np.float32)
    B = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            A[i][j] = np.exp(i) * np.sin(j)

    local_n = n // size
    local_start = rank * local_n
    local_end = local_start + local_n

    start_time = time.time()

    for i in range(local_start, local_end):
        for j in range(m):
            B[i][j] = np.exp(i) * np.sin(j)

    end_time = time.time()

    result = comm.gather(B[local_start:local_end], root=0)

    if rank == 0:
        final_result = np.concatenate(result)
        print(f"Array size {n}x{n}: Time - {end_time - start_time} sec. Result {B}")
