#include "FuzzyLogic.hpp"

double Not(double x) {
	return 1 - x;
}

double Not2(double x) {
	return sqrt(1 - pow(x, 2));
}

double Not3(double x) {
	return pow(1 - pow(x, 3), 1.0 / 3);
}

double And(double x, double y) {
	return x < y ? x : y;
}

double And2(double x, double y) {
	return x * y;
}

double Or(double x, double y) {
	return x > y ? x : y;
}

double Or2(double x, double y) {
	return ( x + y ) - ( x * y );
}