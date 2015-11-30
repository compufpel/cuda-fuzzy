/* CudaFuzzy Project - 2015 
 * Graduate Program in Computer Science - UFPel
 *
 * \file d_BulkLogic.cu -- Device Bulk Logic
 *      This file contains the parallel implementations of fuzzy functions applied to arrays.
 */

#include "d_BulkLogic.cuh"

/*___kernel_Not___
*   Function: Implement the kernel of parallel version of Fuzzy not operation (1-Input).
*   Parameters:
*   Input:  double* array: Input of Basic Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_Not(double* array, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Not(array[idx]);
	}
}


/*___d_BulkNot___ 
*   Function: Implement the parallel version of Fuzzy not operation (1-Input).
*   Parameters:
*   Input:  double* v: Array Fuzzy that will be operated 
*           int size : Number of array elements
*   Output: double* result: Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkNot(double* v, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);

	kernel_Not<<< size, size >>>(d_array, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

/*___kernel_Not2___
*   Function: mplement the kernel of parallel version of Fuzzy not operation sqrt(1 - pow(x, 2)).
*   Parameters:
*   Input:  double* array: Input of Basic Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_Not2(double* array, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Not2(array[idx]);
	}
}

/*___d_BulkNot2___
*   Function: Implement the parallel version of Fuzzy not operation sqrt(1 - pow(v[n], 2)) with Fuzzy array.
*   Parameters:
*   Input:  double* v: Array Fuzzy that will be operated 
*           int size : Number of array elements
*   Output: double* result: Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkNot2(double* v, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);

	kernel_Not2<<< size, size >>>(d_array, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

/*___kernel_Not3___
*   Function: Implement the kernel of parallel version of Fuzzy not operation pow(1 - pow(x, 3), 1.0 / 3).
*   Parameters:
*   Input:  double* array: Input of Basic Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_Not3(double* array, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Not3(array[idx]);
	}
}

/*___d_BulkNot3___
*   Function: Implement the parallel version of Fuzzy not operation pow(1 - pow(v[n], 3), 1.0 / 3) with Fuzzy array.
*   Parameters:
*   Input:  double* v: Array Fuzzy that will be operated 
*           int size : Number of array elements
*   Output: double* result: Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkNot3(double* v, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);

	kernel_Not3<<< size, size >>>(d_array, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

/*___kernel_And___
*   Function: Implement the kernel of parallel version of Fuzzy and operation (<).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_And(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_And(array[idx], array2[idx]);
	}
}

/*___d_BulkAnd___
*   Function: Implement the parallel version of Fuzzy not operation (<) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array1 Fuzzy that will be operated 
*          double* w: Array2 Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkAnd(double* v, double* w, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 
	double* d_array2; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	cudaMalloc((void**) &d_array2, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice);

	kernel_And<<< size, size >>>(d_array, d_array2, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

/*___kernel_And2___
*   Function: Implement the kernel of parallel version of Fuzzy And operation (*).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_And2(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_And2(array[idx], array2[idx]);
	}
}

/*___d_BulkAnd2___
*   Function: Implement the parallel version of Fuzzy And2 operation (*) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array1 Fuzzy that will be operated 
*          double* w: Array2 Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkAnd2(double* v, double* w, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 
	double* d_array2; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	cudaMalloc((void**) &d_array2, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice);

	kernel_And<<< size, size >>>(d_array, d_array2, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

/*___kernel_Or___
*   Function: Implement the kernel of parallel version of Fuzzy Or operation (>).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_Or(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Or(array[idx], array2[idx]);
	}
}

/*___d_BulkOr___
*   Function: Implement the parallel version of Fuzzy Or operation (>) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array1 Fuzzy that will be operated 
*          double* w: Array2 Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkOr(double* v, double* w, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 
	double* d_array2; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	cudaMalloc((void**) &d_array2, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice);

	kernel_Or<<< size, size >>>(d_array, d_array2, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

/*___kernel_Or2___
*   Function: Implement the kernel of parallel version of Fuzzy Or2 operation (( x + y ) - ( x * y )).
*   Parameters:
*   Input:  double* array1: Input of Basic And Fuzzy operation
*           double* array2: Input of Basic And Fuzzy operation
*           int size : Size of vector
*   Output: double* result: Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
__global__ void kernel_Or2(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Or2(array[idx], array2[idx]);
	}
}

/*___d_BulkOr2___
*   Function: Implement the parallel version of Fuzzy Or2 operation (( v[n] + w[n] ) - ( v[n] * w[n] )) with Fuzzy array.
*   Parameters:
*   Input: double* v: Array1 Fuzzy that will be operated 
*          double* w: Array2 Fuzzy that will be operated 
*          int size : Number of array elements
*   Output: double* : Operated Array Fuzzy
*   Creation date: November, 2015.
*   Exception case: -
*/
double* d_BulkOr2(double* v, double* w, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);
	double* d_result; 
	double* d_array; 
	double* d_array2; 

	unsigned int memsize = sizeof(double) * size;

	cudaMalloc((void**) &d_result, memsize);
	cudaMalloc((void**) &d_array, memsize);
	cudaMalloc((void**) &d_array2, memsize);
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice);

	kernel_Or2<<< size, size >>>(d_array, d_array2, d_result, size);

	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

	return result;
}

