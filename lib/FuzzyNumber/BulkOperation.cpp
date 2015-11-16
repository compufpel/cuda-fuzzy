#include "BulkOperation.hpp"

BulkOperation::BulkOperation (vector<Operation*> operations) {
	
	this->operations = operations;
	
}
		
vector<FuzzyNumber*> BulkOperation::execute() {
	
	vector<FuzzyNumber*> result(this->operations.size());
	
	for (int i = 0; i < this->operations.size(); i++) {
		result[i] = this->operations[i]->execute();
	}
	
	return result;
	
}