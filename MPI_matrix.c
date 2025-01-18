#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <stdbool.h>
#include <mpi.h>


#define NUM_RUNS 10


// Random to apply random float numbers 
float randomFloat_Matrix(float min, float max){
    return min + ((float)rand() / RAND_MAX) * (max - min);
}


// Check Symmetry no Optimization
void checkSymPrint(float* matrix, int N) {

    int isSym = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {  // Only check the lower triangle
            if (matrix[i * N + j] != matrix[j * N + i]) {
                isSym = 0;
            }
        }
    }
    if (isSym)
    {
        printf("SERIAL_Matrix is symmetric.\n");       
    }else{
        printf("SERIAL_Matrix is not symmetric.\n");
    }
}

// Check symmetry no Optimization
bool checkSym(float* matrix, int N) {
    int isSym = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            if (matrix[i * N + j] != matrix[j * N + i]) {
                isSym = false;
            }
        }
    }
    return isSym;
}

// Transpose for no Optimization
void matTranspose(float* matrix, float* transposed_matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            transposed_matrix[j * N + i] = matrix[i * N + j];
        }
    }
}

// Check Symmetry using OpenMP
bool checkSymOpenMP(float* matrix, int N, int threads) {
    int isSym = true;
    // Parallelize the loop with OpenMP
    #pragma omp parallel for /* collapse(2) */ /*schedule(dynamic)*/ shared(matrix, N, threads) num_threads(threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {  // Only check the lower triangle
            if (matrix[i * N + j] != matrix[j * N + i]) {
                isSym = false;
            }
        }
    }
    return isSym;
}


// Matrix Transposition using OpenMP
void matTransposeOpenMP(float* matrix, float* transposed_matrix, int N, int threads) {
    #pragma omp parallel for collapse(2) /*schedule(dynamic)*/ shared(matrix, transposed_matrix, N, threads) num_threads(threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            transposed_matrix[j * N + i] = matrix[i * N + j];
        }
    }
}


int main(int argc, char* argv[]) {

    srand(time(NULL));

    float min = 1.0;
    float max = 100.0;
    double start_time, end_time;
    
    int rank, size;
    
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Check if the command-line argument is provided
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1; // Exit if the argument is missing
    }
    
    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Error: Matrix dimension must be a positive integer.\n");
        return 1; // Exit if the input is invalid
    }
    
    
    // Allocate memory for the matrix and transposed matrix on all processes
    float* matrix = (float*)malloc(N * N * sizeof(float));
    float* transposed_matrix = (float*)malloc(N * N * sizeof(float));
    
    // Initialize the matrix
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = randomFloat_Matrix(min,max);
            }
        }

//-----------------------------------------------------------------------------------------


    
//-----------------------------------------------------------------------------------------
// SERIAL CODE FOR SYMMETRY AND TRANSPOSITION

    if(rank == 0){
    //Print if symmetryc
    checkSymPrint(matrix,N);
    }
    
    double avg_elapsed_time_serial_sym = 0.0;
    double avg_elapsed_time_serial_trans = 0.0;

if (rank == 0) {
    // =======================
    // 1.1 Symmetry
    // =======================
    bool isSym = true;
    double elapsed_time_serial_sym = 0.0;

    for (int t = 0; t < NUM_RUNS; t++) {
        start_time = MPI_Wtime();
        isSym = checkSym(matrix, N);
        end_time = MPI_Wtime();
        elapsed_time_serial_sym += (end_time - start_time);
    }

    // Average over NUM_RUNS
    avg_elapsed_time_serial_sym = elapsed_time_serial_sym / NUM_RUNS;
    printf("Time taken for the checkSym: %.6f seconds and the matrix symmetry is: %s\n", 
           avg_elapsed_time_serial_sym, isSym ? "true" : "false");

    // =======================
    // 1.4 matTranspose
    // =======================
    double elapsed_time_serial_trans = 0.0;

    for (int t = 0; t < NUM_RUNS; t++) {
        start_time = MPI_Wtime();
        matTranspose(matrix, transposed_matrix, N);
        end_time = MPI_Wtime();
        elapsed_time_serial_trans += (end_time - start_time);
    }

    // Average over NUM_RUNS
    avg_elapsed_time_serial_trans = elapsed_time_serial_trans / NUM_RUNS;
    printf("Time taken for the matTranspose: %.6f seconds\n", avg_elapsed_time_serial_trans);
    printf("\n");

    // Open CSV file for writing results
    FILE *file1 = fopen("SERIAL.csv", "a");
    if (file1 == NULL) {
        perror("Failed to open file");
        return 1;
    }

    // Write header for CSV SERIAL
    fprintf(file1, "\n SERIAL %d \n", N);
    fprintf(file1, "Operation, Average Time (s)\n");

    // Write results for symmetry and transpose operations
    fprintf(file1, "checkSym, %.6f\n", avg_elapsed_time_serial_sym);
    fprintf(file1, "matTranspose, %.6f\n", avg_elapsed_time_serial_trans);

    fclose(file1);
}


////-----------------------------------------------------------------------------------------
//// OMP PARALLEL CODE FOR SYMMETRY AND TRANSPOSITION
//
//if (rank == 0) {
//
//    // Open CSV file for writing results
//    FILE *file2 = fopen("OMP.csv", "a");
//    if (file2 == NULL) {
//        perror("Failed to open file");
//        return 1;
//    }
//    
//    // =======================
//    // 1.3 checkSymOpenMP
//    // =======================
//
//    // Write header for CSV OPENMP
//    fprintf(file2, "\n OPENMP %d \n", N);
//    fprintf(file2, "threads, Sym_OpenMP, bandwidth, Speedup, Efficiency\n");
//
//    // Test performance with thread counts of 1, 2, 4, 8, 16, 32, and 64
//    for (int threads = 1; threads <= 64; threads *= 2) {
//        omp_set_num_threads(threads);
//
//        double total_time_parallel = 0.0;
//        for (int run = 0; run < NUM_RUNS; run++) {
//            start_time = MPI_Wtime();
//            checkSymOpenMP(matrix, N, threads);
//            end_time = MPI_Wtime();
//            total_time_parallel += (end_time - start_time);
//        }
//
//        double avg_time_parallel = total_time_parallel / NUM_RUNS;
//        size_t total_memory_accessed1 = 2 * N * N * sizeof(float);
//        double bandwidth1 = total_memory_accessed1 / (avg_time_parallel * 1e9);
//        double avg_speedup_sym = avg_elapsed_time_serial_sym / avg_time_parallel;
//        double avg_efficiency_sym = (avg_speedup_sym / threads) * 100;
//        
//        printf("Time taken for the checkSymOpenMP: %.6f seconds, threads: %d\n", avg_time_parallel, threads);
//        printf("bandwidth, Speedup, Efficiency: %.4f GB/s, %.6f, %.2f%% \n", bandwidth1,avg_speedup_sym,avg_efficiency_sym);
//
//        // Write results to CSV
//        fprintf(file2, "%d, %.6f, %.4f, %.6f, %.2f\n",
//                threads, avg_time_parallel, bandwidth1, avg_speedup_sym, avg_efficiency_sym);
//    }
//    printf("\n");
//
//    // =======================
//    // 1.6 matTransposeOpenMP
//    // =======================
//    
//    fprintf(file2, "threads, Tran_OpenMP, bandwidth, Speedup, Efficiency\n");
//
//    for (int threads = 1; threads <= 64; threads *= 2) {
//        double total_time_parallel_transpose = 0.0;
//        for (int run = 0; run < NUM_RUNS; run++) {
//            start_time = MPI_Wtime();
//            matTransposeOpenMP(matrix, transposed_matrix, N, threads);
//            end_time = MPI_Wtime();
//            total_time_parallel_transpose += (end_time - start_time);
//        }
//
//        double avg_time_parallel_transposta = total_time_parallel_transpose / NUM_RUNS;
//        size_t total_memory_accessed2 = 2 * N * N * sizeof(float);
//        double bandwidth2 = total_memory_accessed2 / (avg_time_parallel_transposta * 1e9);
//        double avg_speedup_t = avg_elapsed_time_serial_trans / avg_time_parallel_transposta;
//        double avg_efficiency_t = (avg_speedup_t / threads) * 100;
//
//        printf("Time taken for the matTransposeOpenMP: %.6f seconds, threads: %d\n", avg_time_parallel_transposta, threads);
//        printf("bandwidth, Speedup, Efficiency: %.4f GB/s, %.6f, %.2f%% \n", bandwidth2,avg_speedup_t,avg_efficiency_t);
//
//
//        // Write results to CSV
//        fprintf(file2, "%d, %.6f, %.4f, %.6f, %.2f\n",
//                threads, avg_time_parallel_transposta, bandwidth2, avg_speedup_t, avg_efficiency_t);
//    }
//
//    fclose(file2);
//    printf("\n");
//}




//----------------------------------------------------------------------------------------------
    
    
    // PARALLEL CODE FOR TRANSPOSITION AND SYMMETRY WITH MPI
    

    // Ensure matrix size is divisible by number of processes
    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: Matrix size N must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        exit(1);
    }
    
    
    // Initialize the matrix
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = randomFloat_Matrix(min,max);
            }
        }
    }

//------------------------------------------------------------------------------------------

  //=================================
  // Matrix Symmetry Check using MPI
  //=================================
    
  // Broadcast the matrix to all processes
    MPI_Bcast(matrix, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Divide the rows of the matrix between the processes
    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : (rank + 1) * rows_per_process;
    bool local_symmetry = true;
    
    double sym_total_computation_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
      double start_time = MPI_Wtime();
      // Check the symmetry for the assigned rows
      for (int i = start_row; i < end_row; i++) {
          for (int j = 0; j < N; j++) {
              // Check symmetry: matrix[i][j] should be equal to matrix[j][i]
              if (matrix[i * N + j] != matrix[j * N + i]) {
                  local_symmetry = false;
              }
          }
      }
      double end_time = MPI_Wtime();
      sym_total_computation_time += (end_time - start_time); 
    }
    
    // Average computation time
    double sym_avg_computation_time = sym_total_computation_time / NUM_RUNS;
    double sym_weak_scaling_MPI = sym_avg_computation_time; // same proportion trow different processors number and matrix sizes
    double sym_speedup_MPI = avg_elapsed_time_serial_sym / sym_avg_computation_time;
    double sym_efficiency_MPI = (sym_speedup_MPI / size) * 100;
    

    // Gather the results of the symmetry check from all processes at the root
    bool* all_symmetry_results = NULL;
    if (rank == 0) {
        all_symmetry_results = (bool*)malloc(size * sizeof(bool));
    }

    MPI_Gather(&local_symmetry, 1, MPI_C_BOOL, all_symmetry_results, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // If rank 0, check if all results are true
    if (rank == 0) {
        bool is_symmetric = true;
        for (int i = 0; i < size; i++) {
            if (!all_symmetry_results[i]) {
                is_symmetric = false;
                break;
            }
        }
        printf("MPI_The matrix is symmetric?. %s\n", is_symmetric ? "true" : "false");       
    }
    free(all_symmetry_results); // free memory

    // Bandwidth for Symmetry MPI check
    size_t total_memory_accessed3 = 2 * N * N * sizeof(float); // Read + Write
    double sym_bandwidth3 = total_memory_accessed3 / (sym_avg_computation_time * 1e9);
    

    //===============================================================================================    
    // Computaion time, Strong scaling / speedup, Weak scaling,  Efficency for SYMMETRY CHECK with MPI
    //===============================================================================================

  // Display the average computation time
  if (rank == 0) {
  
    printf("MPI version with processors = %d\n", size);
    printf("Matrix dimension: %d\n", N);
		printf("Running each simulation %d times\n", NUM_RUNS);
    printf("Average Computation Time over %d runs: %f seconds\n", NUM_RUNS, sym_avg_computation_time);
    printf("Weak Scaling Time : %f seconds\n", sym_weak_scaling_MPI);
    printf("Speedup Time: %7.2f seconds\n", sym_speedup_MPI);
    printf("Efficiency Time: %9.2f%%\n", sym_efficiency_MPI);
    printf("Bandwidth: %.4f GB/s \n", sym_bandwidth3);
    printf("\n");
    
    // Open CSV file for writing results
    FILE *file3 = fopen("SYMMETRY_MPI.csv", "a");
    if (file3 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    
    // Write header for CSV MPI
    fprintf(file3, "\n N, Processors,sym_avg_computation_time,sym_weak_scaling_MPI,sym_speedup_MPI,sym_efficiency_MPI,sym_bandwidth \n");

    fprintf(file3, "%d,%d,%f,%f,%f,%.2f,%.4f\n",N,size,sym_avg_computation_time,sym_weak_scaling_MPI,sym_speedup_MPI,sym_efficiency_MPI,sym_bandwidth3);
    fclose(file3);
    }    
    
// -----------------------------------------------------------------------------------------------------------    
    
    //===================================
    // Matrix Transposition using MPI
    //===================================
    

    // Broadcast the transposed data to the different processors
    MPI_Bcast(matrix, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    rows_per_process = N / size;
    float* local_transposed = (float*)malloc(N * rows_per_process * sizeof(float));
    double total_computation_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        // Measure computation time for the transposition NUM_RUNS times
        double start_time = MPI_Wtime();

        for (int i = rank * rows_per_process; i < (rank + 1) * rows_per_process; i++) {
            for (int j = 0; j < N; j++) {
                local_transposed[(j * rows_per_process + i % rows_per_process)] = matrix[i * N + j];
            }
        }

        double end_time = MPI_Wtime();
        total_computation_time += (end_time - start_time); 
    }

    // Average computation time
    double avg_computation_time = total_computation_time / NUM_RUNS;
    double weak_scaling_MPI = avg_computation_time; // same proportion trow different processors number and matrix sizes 
    double speedup_MPI = avg_elapsed_time_serial_trans / avg_computation_time;
    double efficiency_MPI = (speedup_MPI / size) * 100;
    
    
    // Gather the transposed data in the correct order
    for (int i = 0; i < N; i++) {
        MPI_Gather(local_transposed + i * rows_per_process, rows_per_process, MPI_FLOAT, transposed_matrix + i * N, rows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    free(local_transposed); // free memory 
    

    //Calculate bandwidth for matTranspose_MPI
    size_t total_memory_accessed4 = 2 * N * N * sizeof(float); // Read + Write
    double t_bandwidth4 = total_memory_accessed4 / (avg_computation_time * 1e9);


  //===============================================================================================
  // Computaion time, Strong scaling / speedup, Weak scaling,  Efficency for TRANSPOSITION with MPI
  //===============================================================================================

  // Display the average computation time
  if (rank == 0) {
    printf("MPI version with processors = %d\n", size);
    printf("Matrix dimension: %d\n", N);
		printf("Running each simulation %d times\n", NUM_RUNS);
    printf("Average Computation Time over %d runs: %f seconds\n", NUM_RUNS, avg_computation_time);
    printf("Weak Scaling Time : %f seconds\n", weak_scaling_MPI);
    printf("Speedup Time: %7.2f seconds\n", speedup_MPI);
    printf("Efficiency Time: %.2f%%\n", efficiency_MPI);
    printf("Bandwidth: %.4f GB/s \n", t_bandwidth4);
    printf("\n");



    // Open CSV file for writing results
    FILE *file4 = fopen("TRANSPOSE_MPI.csv", "a");
    if (file4 == NULL) {
        perror("Failed to open file");
        return 1;
    }
    
    // Write header for CSV Transposition using MPI
    fprintf(file4, "\n N,Processors,avg_computation_time,weak_scaling_MPI,speedup_MPI,efficiency_MPI,t_bandwidth \n");

    fprintf(file4, "%d,%d,%f,%f,%f,%.2f,%.4f \n",N,size,avg_computation_time,weak_scaling_MPI,speedup_MPI,efficiency_MPI,t_bandwidth4);
    fclose(file4);
    }
  
//-------------------------------------------------------------------------------------------------------------------

    
    
//    // Only the root process (rank 0) prints the original matrix and  the transposed matrix
//    if (rank == 0) {
//    
//    // Only the root process (rank 0) prints the original matrix
//    printf("Original matrix:\n");
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++) {
//                printf("%f ", matrix[i * N + j]);
//            }
//            printf("\n");
//        }
//    
//    
//    
//    printf("\nTransposed matrix:\n");
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++) {
//                printf("%f ", transposed_matrix[i * N + j]);
//            }
//            printf("\n");
//        }
//    }
 

    
    // Free allocated memory
    free(matrix);
    free(transposed_matrix);
    
    
    
    

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
