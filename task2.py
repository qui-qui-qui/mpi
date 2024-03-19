from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

array_sizes = [10, 1000, 10000000]

for ar_size in array_sizes:
    data = None

    if rank == 0:
        data = np.arange(ar_size)
    
    data = comm.bcast(data, root=0)

    start_time = time.time()

    N = len(data)
    size = comm.Get_size()

    sendcounts = np.array([N // size] * size)
    sendcounts[:N % size] += 1
    displacements = np.hstack((0, np.cumsum(sendcounts)[:-1]))

    recvbuf = np.zeros(sendcounts[rank], dtype=np.uint64)

    comm.Scatterv([data, sendcounts, displacements, MPI.INT], recvbuf, root=0)

    local_sum = np.sum(recvbuf)
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"Sum of {ar_size} elements: {global_sum}, Execution time: {end_time - start_time} sec.")
