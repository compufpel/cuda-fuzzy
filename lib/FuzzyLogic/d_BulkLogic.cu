#include "d_BulkLogic.cuh"

__global__ void kernel_Not(double* array, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Not(array[idx]);
	}
}

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

__global__ void kernel_Not2(double* array, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Not2(array[idx]);
	}
}

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

__global__ void kernel_Not3(double* array, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Not3(array[idx]);
	}
}

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

__global__ void kernel_And(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_And(array[idx], array2[idx]);
	}
}

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

__global__ void kernel_And2(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_And2(array[idx], array2[idx]);
	}
}

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

__global__ void kernel_Or(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Or(array[idx], array2[idx]);
	}
}

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

__global__ void kernel_Or2(double* array, double* array2, double* result, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size) {
		result[idx] = d_Or2(array[idx], array2[idx]);
	}
}

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

