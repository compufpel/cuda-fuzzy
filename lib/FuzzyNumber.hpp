#ifndef fuzzy_number
#define fuzzy_number

class FuzzyNumber {

	private:
	
		float begin, end;
	
	public:
		FuzzyNumber (float begin, float end);
		float Begin();
		float End();
		FuzzyNumber* operator+ (FuzzyNumber other);
		FuzzyNumber* operator- (FuzzyNumber other);
		FuzzyNumber* operator* (FuzzyNumber* other);
		FuzzyNumber* operator/ (FuzzyNumber* other);
	
};

#endif