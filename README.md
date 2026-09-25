# Parallel & GPU Computing

This repository contains practical implementations, source code, and experimental results for the **Parallel & GPU Computing** course. It focuses on analyzing and comparing the computational performance of matrix operations across different execution models, starting with a single-core sequential baseline and progressing to multi-core shared-memory parallelization.

---

## Experiments

### Experiment 1: Matrix Multiplication

| Part | Experiment | Implementation | Documentation |
| :--- | :--- | :--- | :--- |
| **Part A** | Matrix Multiplication | Sequential (Baseline) | [Part A - Sequential README](Experiment-1/Part-A-Sequential/README.md) |
| **Part B** | Matrix Multiplication | OpenMP (Shared Memory) | [Part B - OpenMP README](Experiment-1/Part-B-OpenMP/README.md) |

---

## Repository Structure

```text
Parallel-GPU-Computing/
│
├── README.md
│
└── Experiment-1/
    ├── Part-A-Sequential/
    │   ├── README.md
    │   ├── code/
    │   │   └── sequential_matrix.c
    │   └── screenshots/
    │       ├── Sequential Matrix Output 1.png
    │       └── ... (9 images)
    │
    └── Part-B-OpenMP/
        ├── README.md
        ├── code/
        │   └── openmp_matrix.c
        └── screenshots/
            ├── OpenMP Matrix Output 1.png
            └── ... (5 images)
```

---

## Technologies Used

* **C** (Standard C99/C11)
* **OpenMP** (Open Multi-Processing API)
* **GCC** (GNU Compiler Collection)
