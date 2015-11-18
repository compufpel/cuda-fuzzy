#ifndef d_fuzzy_logic
#define d_fuzzy_logic

inline __device__ double d_Not(double x) {
	return 1 - x;
}

inline __device__ double d_Not2(double x) {
	return sqrt(1 - pow(x, 2));
}

inline __device__ double d_Not3(double x) {
	return pow(1 - pow(x, 3), 1.0 / 3);
}

inline __device__ double d_And(double x, double y) {
	return x < y ? x : y;
}

inline __device__ double d_And2(double x, double y) {
	return x * y;
}

inline __device__ double d_Or(double x, double y) {
	return x > y ? x : y;
}

inline __device__ double d_Or2(double x, double y) {
	return ( x + y ) - ( x * y );
}

#endif