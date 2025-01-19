# IntroPARCO-2024-H2
Here there are the information to reproducing the results:

The code is one so to obtain the different results is needed to comment and uncomment the the desired parts.

For compilation and running operation I use 1 .pbs
1. mpiC.pbs for running all the code

The result are all printed in the relative .cvs files

Processor: Intel(R) Core(TM) i7-8565U CPU 1.80Hz, 4 cores, 8 logical processors.

RAM: 8GB DDR4.

Operating System: Windows 11

Compiler: GCC 13.2.0, G++ 13.2.0 (MinGW Compiler), GCC 9.1.0 on the cluster with MPI support.

All my training runs was done on an interractive section due to impossibility of compiling in my own system:

qsub -q short_cpuQ -I -l select=1:ncpus=16:mpiprocs=16:mem=1000mb

module load mpich-3.2.1--gcc-9.1.0

Libraries: 

#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#include <sys/time.h>

#include <unistd.h>

#include <omp.h>

#include <stdbool.h>

#include <mpi.h>
