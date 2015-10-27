#include "FuzzyNumber.hpp"

FuzzyNumber::FuzzyNumber (float begin, float end) {

	this->begin = begin;
	this->end = end;
	
}

float FuzzyNumber::Begin() {
	return this->begin;
}

float FuzzyNumber::End() {
	return this->end;
}

FuzzyNumber* FuzzyNumber::operator+ (FuzzyNumber other) {

	return new FuzzyNumber(this->Begin() + other.Begin(), this->End() + other.End());
	
}

FuzzyNumber* FuzzyNumber::operator- (FuzzyNumber other) {

	return new FuzzyNumber(this->Begin() - other.End(), this->End() - other.Begin());
	
}




