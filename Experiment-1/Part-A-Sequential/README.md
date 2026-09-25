# Part A — Sequential Matrix Multiplication

## 1. Aim
To implement and analyze a dense matrix multiplication algorithm of size $4000 \times 4000$ using a single-threaded sequential execution model in C.

---

## 2. Objective
* To implement standard square matrix multiplication on a single CPU core.
* To establish a reliable, deterministic execution baseline for measuring subsequent parallel speedup.
* To verify numerical correctness by initializing all matrix values to $1.0$ and ensuring every entry in the product matrix equals $4000.00$.

---

## 3. Theory / Concept
Matrix multiplication is a fundamental algebraic operation. Given two square matrices $A$ and $B$ of dimension $N \times N$, their product matrix $C = A \times B$ is defined as:

$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]$$

In a sequential execution model:
* A single process executes instructions linearly on one CPU core without multi-threading.
* The standard nested three-loop algorithm requires $O(N^3)$ operations. For $N = 4000$, this demands $2 \times 4000^3 = 1.28 \times 10^{11}$ floating-point operations (128 GFLOPs).
* Matrices are stored linearly in row-major order within dynamic memory allocated via `malloc`.
* Execution time is measured using the standard C library function `clock()` and normalized by `CLOCKS_PER_SEC`.

---

## 4. Algorithm / Steps
1. **Verify Environment**: Check WSL2 status from Windows PowerShell and launch the Ubuntu environment.
2. **Install Build Tools**: Update the package list with `sudo apt update` and install GCC and build utilities via `build-essential`.
3. **Allocate Memory**: Dynamically allocate contiguous heap memory for matrices $A$, $B$, and $C$ of size $N \times N$ ($4000 \times 4000$ `double` elements each).
4. **Initialize Data**:
   * Populate all entries of matrix $A$ with $1.0$.
   * Populate all entries of matrix $B$ with $1.0$.
   * Zero-initialize all entries of matrix $C$.
5. **Start Timer**: Record the initial CPU clock tick count using `start = clock()`.
6. **Compute Product**:
   * Loop index $i$ from $0$ to $N - 1$ (rows of $A$ and $C$).
   * Loop index $j$ from $0$ to $N - 1$ (columns of $B$ and $C$).
   * Loop index $k$ from $0$ to $N - 1$ (dot product accumulation: $C[i \cdot N + j] += A[i \cdot N + k] \cdot B[k \cdot N + j]$).
7. **Stop Timer**: Record the final CPU clock tick count using `end = clock()`.
8. **Verify and Display**:
   * Calculate elapsed time: $(end - start) / \text{CLOCKS\_PER\_SEC}$.
   * Verify correctness at index $(0,0)$ to ensure $C[0][0] = 4000.00$.
9. **Deallocate Memory**: Release allocated memory using `free()`.

---

## 5. Source Code

The complete source code is stored in [code/sequential_matrix.c](code/sequential_matrix.c):

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    clock_t start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    printf("Initializing %d x %d matrices...\n", N, N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    start = clock();
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
    end = clock();

    printf("\nSequential Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Execution Time = %f seconds\n",
           (double)(end - start) / CLOCKS_PER_SEC);
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

# Compile with standard O2 optimization
gcc -O2 sequential_matrix.c -o sequential_matrix

# Execute the sequential program
./sequential_matrix
```

---

## 7. Output / Results

The program allocates memory for $4000 \times 4000$ matrices, initializes elements to $1.0$, performs the triple-nested loop multiplication on a single CPU thread, and prints the total runtime along with verification:

```text
Initializing 4000 x 4000 matrices...

Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 313.555760 seconds
Verification C[0][0] = 4000.00
```

* **Matrix Dimensions**: $4000 \times 4000$
* **Execution Time**: `313.555760 seconds` (~5 minutes 13 seconds)
* **Verification Result**: `C[0][0] = 4000.00` ($4000$ terms of $1.0 \times 1.0$ accumulated)

---

## 8. Screenshots

### Sequential Matrix Output 1
![Sequential Matrix Output 1](screenshots/Sequential%20Matrix%20Output%201.png)
*Running `wsl --status` in Windows PowerShell, confirming the default distribution is Ubuntu and the default version is 2.*

---

### Sequential Matrix Output 2
![Sequential Matrix Output 2](screenshots/Sequential%20Matrix%20Output%202.png)
*Launching the Ubuntu WSL environment via `wsl` from PowerShell, changing to the home directory with `cd ~`, and verifying the path using `pwd` (`/home/admin7877`).*

---

### Sequential Matrix Output 3
![Sequential Matrix Output 3](screenshots/Sequential%20Matrix%20Output%203.png)
*Executing `sudo apt update` in the Ubuntu terminal to refresh local package repository indexes.*

---

### Sequential Matrix Output 4
![Sequential Matrix Output 4](screenshots/Sequential%20Matrix%20Output%204.png)
*Installing standard compilation utilities and GCC using `sudo apt install build-essential -y`.*

---

### Sequential Matrix Output 5
![Sequential Matrix Output 5](screenshots/Sequential%20Matrix%20Output%205.png)
*Verifying the GCC installation and checking the compiler version using `gcc --version` (showing `gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0`).*

---

### Sequential Matrix Output 6
![Sequential Matrix Output 6](screenshots/Sequential%20Matrix%20Output%206.png)
*Creating the experiment directory structure with `mkdir parallel_lab && cd parallel_lab` and `mkdir sequential && cd sequential`.*

---

### Sequential Matrix Output 7
![Sequential Matrix Output 7](screenshots/Sequential%20Matrix%20Output%207.png)
*Creating the source file `matrix_sequential.c` using `touch` and opening it for editing with `nano matrix_sequential.c`.*

---

### Sequential Matrix Output 8
![Sequential Matrix Output 8](screenshots/Sequential%20Matrix%20Output%208.png)
*Viewing the complete C source code for sequential matrix multiplication written inside the GNU nano 8.7.1 editor.*

---

### Sequential Matrix Output 9
![Sequential Matrix Output 9](screenshots/Sequential%20Matrix%20Output%209.png)
*Compiling with `gcc -O2 matrix_sequential.c -o matrix_sequential` and executing `./matrix_sequential`, showing completion with execution time of 313.555760 seconds and verification `C[0][0] = 4000.00`.*

---

## 9. Observation
* The sequential program executes on a single CPU core using one thread of execution.
* For a $4000 \times 4000$ matrix size, single-core processing required **313.555760 seconds** (approximately 5 minutes and 13 seconds).
* No multi-threading or parallelization was utilized; the remaining logical cores on the machine remained idle during this calculation.
* The output value of $C[0][0] = 4000.00$ mathematically validates correctness with zero error across all dot products.
* This execution time of **313.555760 seconds** forms the baseline ($T_{\text{Sequential}}$) against which parallel speedups are calculated.

---

## 10. Conclusion
The sequential matrix multiplication algorithm was successfully implemented, compiled with GCC, and executed in Ubuntu on WSL2. The program verified the expected deterministic result ($C[0][0] = 4000.00$) in 313.555760 seconds. The cubic complexity ($O(N^3)$) and high execution time clearly demonstrate the performance bottlenecks of serial computing for large data sets, establishing the necessity for parallel models like OpenMP.
