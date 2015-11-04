#ifndef bulk_operation
#define bulk_operation
#include "FuzzyNumber.hpp"
#include "Operation.hpp"
#include <iostream>
#include <vector>

using namespace std;

class BulkOperation {

	private:
	
		vector<Operation*> operations;
	
	public:
	
		BulkOperation (vector<Operation*> operations);
		vector<FuzzyNumber*> execute();
	
};

#endif