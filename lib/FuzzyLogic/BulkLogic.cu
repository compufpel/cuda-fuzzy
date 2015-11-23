#include "BulkLogic.cuh"

/*___h_BulkNot___
*   Objective: Implement the sequential version of not operation (1-Input) with Fuzzy array.
*   Parameters:
*              -- double* v: Array Fuzzy that will be operated 
*              -- int size : Number of array elements
*   Creation date: November, 2015.
*   Exception case: -
*/
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
