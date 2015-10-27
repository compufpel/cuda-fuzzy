#include "FuzzyNumber.hpp"

//gg commit gurizada
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

FuzzyNumber* FuzzyNumber::operator* (FuzzyNumber other) {

	float min = this->Begin() * this->End(), max = min;
	float r2, r3, r4;

	r2 = this->Begin() * other.Begin();

	if(max < r2){ max = r2; }else{ min = r2; }

	r3 = this->Begin() * other.End();

	if(max < r3){ max = r3; }
	if(min > r3){ min = r3; }

	r4 = this->End() * other.End();

	if(max < r4){ max = r4; }
	if(min > r4){ min = r4; }

	return new FuzzyNumber(max, min);
}




