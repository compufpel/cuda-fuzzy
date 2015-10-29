#ifndef operation_header
#define operation_header
#include "FuzzyNumber.hpp"

class Operation {

	private:
	
		char op;
		FuzzyNumber* a;
		FuzzyNumber* b;
	
	public:
	
		Operation (FuzzyNumber* a, FuzzyNumber* b, char op);
		FuzzyNumber* execute();
	
};

#endif