#!/usr/bin/env bash
#SBATCH --job-name=my_mpi_job         # Job name
#SBATCH --output=job_output_%j.log    # Standard output and error log
#SBATCH --nodes=12                     # Number of nodes
#SBATCH --ntasks=12                   # Total number of MPI tasks requested
#SBATCH --cpus-per-task=8             # Number of CPU cores per MPI task
#SBATCH --time=01:00:00               # Time limit hrs:min:sec

echo "1000
4
4
10 0 0 0   5 1
0 10 0 0   5 1
0 0 10 0   5 1
0 0 0 10   5 1
" > input.txt

export OMP_NUM_THREADS=4
mpirun -np 4 ./solution ./input.txt
# for r in seq 1 12; do
#   mpirun -np "$r" ./solution input.txt
# done
