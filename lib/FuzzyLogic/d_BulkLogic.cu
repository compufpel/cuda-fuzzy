/* CudaFuzzy Project - 2015 
 * Graduate Program in Computer Science - UFPel
 *
 * \file d_BulkLogic.cu -- Device Bulk Logic
 *      This file contains the parallel implementations of fuzzy functions applied to arrays.
 */

#include "d_BulkLogic.cuh"

int _threadsPerBlock = 256;

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

	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;
	
	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host 
	double* d_result; // GPU memory result array
	double* d_array;  // GPU memory input array

	unsigned int memsize = sizeof(double) * size; // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize);  // allocates memory on the GPU to transport input data
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice); // send data from Host do Device

	kernel_Not<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_result, size); // Execute Not kernel function 

	cudaDeviceSynchronize(); // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);

	return result; // Return results
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
	
	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;

	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host 
	double* d_result; // GPU memory result array
	double* d_array; // GPU memory input array

	unsigned int memsize = sizeof(double) * size;  // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize); // allocates memory on the GPU to transport input data
	
	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice); // send data from Host do Device

	kernel_Not2<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_result, size); // Execute Not2 kernel function 

	cudaDeviceSynchronize(); // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);

	return result; // Return results
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
	
	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;

	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host 
	double* d_result; // GPU memory result array
	double* d_array;  // GPU memory input array

	unsigned int memsize = sizeof(double) * size; // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize); // allocates memory on the GPU to transport input data
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice); // send data from Host do Device

	kernel_Not3<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_result, size);  // Execute Not3 kernel function 

	cudaDeviceSynchronize(); // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);

	return result; // Return results
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

	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;
	
	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host 
	double* d_result;  // GPU memory result array
	double* d_array;   // GPU memory input1 array
	double* d_array2;  // GPU memory input2 array

	unsigned int memsize = sizeof(double) * size; // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize);  // allocates memory on the GPU to transport input data
	cudaMalloc((void**) &d_array2, memsize); // allocates memory on the GPU to transport input data
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);  // send data from Host do Device
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice); // send data from Host do Device

	kernel_And<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_array2, d_result, size); // Execute And kernel function

	cudaDeviceSynchronize(); // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);
	cudaFree(d_array2);

	return result; // Return results
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

	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;
	
	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host
	double* d_result;  // GPU memory result array
	double* d_array;   // GPU memory input1 array
	double* d_array2;  // GPU memory input2 array

	unsigned int memsize = sizeof(double) * size; // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize);  // allocates memory on the GPU to transport input data
	cudaMalloc((void**) &d_array2, memsize); // allocates memory on the GPU to transport input data
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);  // send data from Host do Device
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice); // send data from Host do Device

	kernel_And<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_array2, d_result, size); // Execute And2 kernel function

	cudaDeviceSynchronize();  // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);
	cudaFree(d_array2);

	return result;  // Return results
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

	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;
	
	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host
	double* d_result; // GPU memory result array
	double* d_array;  // GPU memory input1 array
	double* d_array2; // GPU memory input2 array

	unsigned int memsize = sizeof(double) * size; // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize);  // allocates memory on the GPU to transport input data
	cudaMalloc((void**) &d_array2, memsize); // allocates memory on the GPU to transport input data
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice); // send data from Host do Device
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice);// send data from Host do Device

	kernel_Or<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_array2, d_result, size); // Execute Or kernel function

	cudaDeviceSynchronize(); // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);
	cudaFree(d_array2);

	return result; // Return results
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

	int blocksPerGrid = size % _threadsPerBlock == 0 ? size /_threadsPerBlock : size /_threadsPerBlock + 1;
	
	double* result = (double*)malloc(sizeof(double) * size); // allocates memory in Host
	double* d_result; // GPU memory result array
	double* d_array; // GPU memory input1 array
	double* d_array2; // GPU memory input2 array

	unsigned int memsize = sizeof(double) * size; // size of arrays of double

	cudaMalloc((void**) &d_result, memsize); // allocates memory on the GPU to save results
	cudaMalloc((void**) &d_array, memsize);  // allocates memory on the GPU to transport input data
	cudaMalloc((void**) &d_array2, memsize); // allocates memory on the GPU to transport input2 data
	

	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);  // send data from Host do Device
	cudaMemcpy(d_array2, w, memsize, cudaMemcpyHostToDevice); // send data from Host do Device

	kernel_Or2<<< blocksPerGrid, _threadsPerBlock >>>(d_array, d_array2, d_result, size); // Execute Or kernel function

	cudaDeviceSynchronize(); // Sync with device

	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost); // Copy results to Host

	cudaFree(d_result);
	cudaFree(d_array);
	cudaFree(d_array2);

	return result; // Return results
}

