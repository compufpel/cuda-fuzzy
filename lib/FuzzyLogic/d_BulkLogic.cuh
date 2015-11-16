#ifndef d_bulk_logic
#define d_bulk_logic
#include <cstdlib>
#include <cmath>
#include <vector>
#include "FuzzyLogic.cuh"
#include "d_FuzzyLogic.cu"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_Not(double* array, double* result, int size);
double* d_BulkNot(double* v, int size);

__global__ void kernel_Not2(double* array, double* result, int size);
double* d_BulkNot2(double* v, int size);

__global__ void kernel_Not3(double* array, double* result, int size);
double* d_BulkNot3(double* v, int size);

__global__ void kernel_And(double* array, double* array2, double* result, int size);
double* d_BulkAnd(double* v, double* w, int size);

__global__ void kernel_And2(double* array, double* array2, double* result, int size);
double* d_BulkAnd2(double* v, double* w, int size);

__global__ void kernel_Or(double* array, double* array2, double* result, int size);
double* d_BulkOr(double* v, double* w, int size);

__global__ void kernel_Or2(double* array, double* array2, double* result, int size);
double* d_BulkOr2(double* v, double* w, int size);

#endif