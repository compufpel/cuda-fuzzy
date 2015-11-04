#include "Operation.hpp"

Operation::Operation (FuzzyNumber* a, FuzzyNumber* b, char op) {
	this->a = a;
	this->b = b;
	this->op = op;
}

FuzzyNumber* Operation::execute() {

	switch (op) {
		case '+':
			return *a + *b;
			break;
		case '-':
			return *a - *b;
			break;
		case '*':
			return (*a) * (*b);
			break;
		case '/':
			return *a / *b;
		default:
			break;
	}
	
}