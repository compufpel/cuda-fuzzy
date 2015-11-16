#include "BulkLogic.cuh"

// double* d_BulkNot(double* v, int size) {
	
// 	double* result = (double*)malloc(sizeof(double) * size);
// 	double* d_result; 
// 	double* d_array; 

// 	unsigned int memsize = sizeof(double) * size;

// 	cudaMalloc((void**) &d_result, memsize);
// 	cudaMalloc((void**) &d_array, memsize);
	

// 	cudaMemcpy(d_array, v, memsize, cudaMemcpyHostToDevice);

// 	cuda_Not<<< size, size >>>(d_array, d_result, size);

// 	cudaDeviceSynchronize();

// 	cudaMemcpy(result, d_result, memsize, cudaMemcpyDeviceToHost);

// 	return result;
// }

double* h_BulkNot(double* v, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);


	for (int i = 0; i < size; i++) {
		result[i] = Not(v[i]);
	}

	return result;
}


double* h_BulkNot2(double* v, int size) {
	double* result = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		result[i] = Not2(v[i]);
	}

	return result;
}

double* h_BulkNot3(double* v, int size) {
	double* result = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		result[i] = Not3(v[i]);
	}

	return result;
}

double* h_BulkAnd(double* v, double* w, int size) {
	
	double* result = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		result[i] = And(v[i], w[i]);
	}

	return result;
}

double* h_BulkAnd2(double* v, double* w, int size) {
	double* result = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		result[i] = And2(v[i], w[i]);
	}

	return result;
}
			
double* h_BulkOr(double* v, double* w, int size) {
	double* result = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		result[i] = Or(v[i], w[i]);
	}

	return result;
}

double* h_BulkOr2(double* v, double* w, int size) {
	double* result = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		result[i] = Or2(v[i], w[i]);
	}

	return result;
}

// __global__ void kernel_Not(double* array, double* result, int size) {

// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 	if(idx < size) {
// 		result[idx] = d_Not(array[idx]);
// 	}
// }
// 				