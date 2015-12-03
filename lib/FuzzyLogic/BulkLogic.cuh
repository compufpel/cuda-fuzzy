#ifndef bulk_logic
#define bulk_logic
#include <cstdlib>
#include <cmath>
#include <vector>
#include "FuzzyLogic.cuh"
#include "d_FuzzyLogic.cu"

#include <cuda.h>
#include <cuda_runtime.h>

/*___h_BulkNot___ 
*   Function: Implement the sequential version of not operation (1-Input) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkNot(double* v, int size);

/*___h_BulkNot2___
*   Function: Implement the sequential version of not operation sqrt(1 - pow(v[n], 2)) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkNot2(double* v, int size);

/*___h_BulkNot3___
*   Function: Implement the sequential version of not operation pow(1 - pow(v[n], 3), 1.0 / 3) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkNot3(double* v, int size);

/*___h_BulkAnd___
*   Function: Implement the sequential version of and operation (<) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkAnd(double* v, double* w, int size);

/*___h_BulkAnd2___
*   Function: Implement the sequential version of and operation (*) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkAnd2(double* v, double* w, int size);

/*___h_BulkOr___
*   Function: Implement the sequential version of not operation (>) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkOr(double* v, double* w, int size);

/*___h_BulkOr2___
*   Function: Implement the sequential version of not operation (( v[n] + w[n] ) - ( v[n] * w[n] )) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* h_BulkOr2(double* v, double* w, int size);

#endif