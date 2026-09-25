# Part B — OpenMP Matrix Multiplication

## 1. Aim
To parallelize the $4000 \times 4000$ matrix multiplication operation on a multi-core CPU using OpenMP shared-memory multi-threading and evaluate the performance gain over the sequential baseline.

---

## 2. Objective
* To distribute outer-loop iterations of matrix multiplication across multiple CPU threads using OpenMP directives.
* To utilize multiple logical CPU cores concurrently in a shared memory address space.
* To measure parallel execution time and calculate the speedup achieved over the single-threaded sequential baseline.
* To verify numerical correctness by ensuring the result entry $C[0][0]$ remains $4000.00$.

---

## 3. Theory / Concept

### OpenMP (Open Multi-Processing)
OpenMP is an Application Programming Interface (API) that provides compiler directives, library routines, and environment variables for shared-memory parallel programming in C, C++, and Fortran. It operates on the **fork-join** execution model:
* Execution starts with a single master thread.
* When a parallel directive is encountered, the master thread forks a team of worker threads.
* Iterations of the parallelized loop are divided among these threads to execute concurrently.
* Once the loop completes, threads synchronize at an implicit barrier and join back into the master thread.

### Threads and Shared Memory
In this experiment, all threads execute within the same process address space and access shared heap memory containing matrices $A$, $B$, and $C$. Because data is shared directly in memory, there is no communication overhead or data copying between threads.

### Relevant OpenMP Directives and Functions Used:
* `#pragma omp parallel for private(j, k)`:
  * Creates a team of threads and distributes iterations of the outermost loop ($i$) among them.
  * The `private(j, k)` clause ensures each thread maintains its own independent inner loop iteration counters, preventing data races.
  * The outer loop counter $i$ is implicitly private to each thread.
* `omp_get_wtime()`:
  * A high-resolution wall-clock timer returning elapsed time in seconds.
* `omp_get_max_threads()`:
  * Returns the maximum number of threads available to the OpenMP runtime.
* `OMP_NUM_THREADS`:
  * An environment variable used to set the number of worker threads executed by OpenMP.

---

## 4. Algorithm / Steps
1. **Check Hardware Capacity**: Query available logical CPU cores using `nproc` (16 logical cores detected).
2. **Configure OpenMP Runtime**: Set thread count with `export OMP_NUM_THREADS=8` and verify using `echo $OMP_NUM_THREADS`.
3. **Allocate Shared Memory**: Allocate contiguous heap memory for matrices $A$, $B$, and $C$ of size $4000 \times 4000$ `double` elements each.
4. **Initialize Data**:
   * Populate all entries of matrix $A$ and $B$ with $1.0$.
   * Initialize matrix $C$ to $0.0$.
5. **Start Wall-Clock Timer**: Record the start timestamp using `start = omp_get_wtime()`.
6. **Parallel Loop Execution**:
   * Execute `#pragma omp parallel for private(j, k)` to distribute the outer loop ($i = 0$ to $N - 1$) across the 8 OpenMP threads.
   * Each thread independently computes its assigned subset of rows in matrix $C$ by multiplying rows of $A$ with columns of $B$.
7. **Join Threads and Stop Timer**: Synchronize all threads at the barrier and record the end timestamp using `end = omp_get_wtime()`.
8. **Verify and Display**:
   * Output the number of threads used and the execution time ($end - start$).
   * Verify that $C[0][0] = 4000.00$.
9. **Monitor System Activity**: Install and launch `htop` to observe multi-core CPU utilization across all cores.
10. **Deallocate Resources**: Free dynamically allocated memory using `free()`.

---

## 5. Source Code

The complete source code is stored in [code/openmp_matrix.c](code/openmp_matrix.c):

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    double start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    start = omp_get_wtime();

    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                C[i * N + j] +=
                    A[i * N + k] *
                    B[k * N + j];
            }
        }
    }

    end = omp_get_wtime();

    printf("OpenMP Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Number of Threads Used = %d\n", omp_get_max_threads());
    printf("Execution Time = %f seconds\n", end - start);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
```

---

## 6. Compilation and Execution

Inside the Ubuntu terminal (WSL2):

```bash
# Navigate to the code directory
cd code

# Set the OpenMP thread count to 8
export OMP_NUM_THREADS=8

# Compile with OpenMP runtime support and O2 optimization
gcc -O2 -fopenmp openmp_matrix.c -o openmp_matrix

# Execute the parallel program
./openmp_matrix
```

---

## 7. Output / Results

The program executes using 8 OpenMP threads across the shared matrices, reporting execution time and output verification:

```text
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 8
Execution Time = 90.547932 seconds
Verification C[0][0] = 4000.00
```

### Performance Comparison

| Implementation | Compute Resources | Execution Time | Verification ($C[0][0]$) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential (Part A)** | 1 CPU Core | `313.555760 s` | `4000.00` | **1.00×** (Baseline) |
| **OpenMP (Part B)** | 8 CPU Threads | `90.547932 s` | `4000.00` | **3.46×** |

$$\text{Speedup} = \frac{T_{\text{Sequential}}}{T_{\text{OpenMP}}} = \frac{313.555760}{90.547932} \approx 3.46\times$$

---

## 8. Screenshots

### OpenMP Matrix Output 1
![OpenMP Matrix Output 1](screenshots/OpenMP%20Matrix%20Output%201.png)
*Launching WSL and querying available logical CPU processing cores using `nproc`, showing 16 logical cores available on the system.*

---

### OpenMP Matrix Output 2
![OpenMP Matrix Output 2](screenshots/OpenMP%20Matrix%20Output%202.png)
*Setting the OpenMP thread count environment variable with `export OMP_NUM_THREADS=8` and verifying the configuration using `echo $OMP_NUM_THREADS` (prints `8`).*

---

### OpenMP Matrix Output 3
![OpenMP Matrix Output 3](screenshots/OpenMP%20Matrix%20Output%203.png)
*Navigating to the project directory, opening the source file, compiling with `gcc -O2 -fopenmp openmp.c -o openmp`, and executing `./openmp`, which displays completion in 90.547932 seconds using 8 threads with verification `C[0][0] = 4000.00`.*

---

### OpenMP Matrix Output 4
![OpenMP Matrix Output 4](screenshots/OpenMP%20Matrix%20Output%204.png)
*Installing the interactive process monitor `htop` inside the Ubuntu terminal via `sudo apt install htop -y`.*

---

### OpenMP Matrix Output 5
![OpenMP Matrix Output 5](screenshots/OpenMP%20Matrix%20Output%205.png)
*Running the `htop` interactive monitor, displaying the active utilization dashboard across all 16 CPU cores (0 to 15), memory usage, and running system tasks.*

---

## 9. Observation
* The OpenMP implementation reduced the computation time from **313.555760 seconds** down to **90.547932 seconds**, achieving a **3.46× speedup**.
* The workload was distributed across 8 worker threads while running on a 16-core logical system.
* Because all threads share the same host physical memory, no data duplication or inter-process communication latency occurred.
* The `private(j, k)` clause prevented race conditions on the inner loop indexes, ensuring clean thread isolation.
* The verification output `C[0][0] = 4000.00` confirms full numerical accuracy matching the sequential baseline.

---

## 10. Conclusion
Parallelizing dense matrix multiplication with OpenMP on a multi-core processor yielded a notable performance improvement, reducing execution time from over 5 minutes down to approximately 1.5 minutes (3.46× speedup). The `#pragma omp parallel for` directive demonstrated the effectiveness and simplicity of shared-memory multi-threading for compute-intensive loop structures without changing the core mathematical algorithm.
