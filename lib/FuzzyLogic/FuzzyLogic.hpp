#ifndef fuzzy_logic
#define fuzzy_logic
#include <cmath>

class FuzzyLogic {
	
	public:
	
		FuzzyLogic ();
		double Not(double x);
		double Not2(double x);
		double Not3(double x);

		double And(double x, double y);
		double And2(double x, double y);

		double Or(double x, double y);
		double Or2(double x, double y);

};

#endif