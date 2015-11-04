#include "FuzzyLogic.hpp"

FuzzyLogic::FuzzyLogic() {}

double FuzzyLogic::Not(double x) {
	return 1 - x;
}

double FuzzyLogic::Not2(double x) {
	return sqrt(1 - pow(x, 2));
}

double FuzzyLogic::Not3(double x) {
	return pow(1 - pow(x, 3), 1.0 / 3);
}