#ifndef bulk_logic
#define bulk_logic
#include <cmath>
#include <vector>
#include "FuzzyLogic.hpp"

using namespace std;

class BulkLogic {
	
	private:
		FuzzyLogic *fuzzy;
	
	public:
		
		BulkLogic ();
		vector<double> Not(vector<double> v);
		vector<double> Not2(vector<double> v);
		vector<double> Not3(vector<double> v);

		vector<double> And(vector<double> v, vector<double> w);
		vector<double> And2(vector<double> v, vector<double> w);

		vector<double> Or(vector<double> v, vector<double> w);
		vector<double> Or2(vector<double> v, vector<double> w);

};

#endif