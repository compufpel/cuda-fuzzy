#ifndef fuzzy_logic
#define fuzzy_logic
#include <cstdlib>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

double Not(double x);
double Not2(double x);
double Not3(double x);

double And(double x, double y);
double And2(double x, double y);

double Or(double x, double y);
double Or2(double x, double y);


#endif