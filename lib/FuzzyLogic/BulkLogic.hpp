#ifndef bulk_logic
#define bulk_logic
#include <cstdlib>
#include <cmath>
#include <vector>
#include "FuzzyLogic.hpp"

//#include <cuda.h>
//#include <cuda_runtime.h>

using namespace std;

//__global__ void CudaNot(double* array, double* result, int size);

double* h_BulkNot(double* v, int size);
double* h_BulkNot2(double* v, int size);
double* h_BulkNot3(double* v, int size);

double* h_BulkAnd(double* v, double* w, int size);
double* h_BulkAnd2(double* v, double* w, int size);

double* h_BulkOr(double* v, double* w, int size);
double* h_BulkOr2(double* v, double* w, int size);

#endif