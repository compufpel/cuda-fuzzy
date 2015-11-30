/* CudaFuzzy Project - 2015 
 * Graduate Program in Computer Science - UFPel - Federal Univesity of Pelotas
 *
 * \file d_BulkLogic.cu -- Device Bulk Logic
 *      This file contains the parallel implementations of fuzzy functions applied to arrays.
 */

#ifndef d_fuzzy_logic
#define d_fuzzy_logic

/*___d_Not___
*   Function: Device Implement of parallel version of Fuzzy Not operation (1-x).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_Not(double x) {
	return 1 - x;
}

/*___d_Not2___
*   Function: Device Implement of parallel version of Fuzzy Not2 operation  sqrt(1 - pow(x, 2)).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_Not2(double x) {
	return sqrt(1 - pow(x, 2));
}

/*___d_Not3___
*   Function: Device Implement of parallel version of Fuzzy Not3 pow(1 - pow(x, 3), 1.0 / 3).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_Not3(double x) {
	return pow(1 - pow(x, 3), 1.0 / 3);
}

/*___d_And___
*   Function: Device Implement of parallel version of Fuzzy Not operation (<).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_And(double x, double y) {
	return x < y ? x : y;
}

/*___d_And2__
*   Function: Device Implement of parallel version of Fuzzy And2 operation (*).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_And2(double x, double y) {
	return x * y;
}

/*___d_Or__
*   Function: Device Implement of parallel version of Fuzzy Or operation (>).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_Or(double x, double y) {
	return x > y ? x : y;
}

/*___d_Or2__
*   Function: Device Implement of parallel version of Fuzzy Or2 operation ( x + y ) - ( x * y ).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
inline __device__ double d_Or2(double x, double y) {
	return ( x + y ) - ( x * y );
}

#endif