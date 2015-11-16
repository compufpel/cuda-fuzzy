#include "BulkLogic.hpp"

BulkLogic::BulkLogic () {
	this->fuzzy = new FuzzyLogic();
}

vector<double> BulkLogic::Not(vector<double> v) {
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->Not(v[i]);
	}

	return result;
}

vector<double> BulkLogic::Not2(vector<double> v) {
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->Not2(v[i]);
	}

	return result;
}

vector<double> BulkLogic::Not3(vector<double> v) {
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->Not3(v[i]);
	}

	return result;
}

vector<double> BulkLogic::And(vector<double> v, vector<double> w) {
	
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->And(v[i], w[i]);
	}

	return result;
}

vector<double> BulkLogic::And2(vector<double> v, vector<double> w) {
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->And2(v[i], w[i]);
	}

	return result;
}
			
vector<double> BulkLogic::Or(vector<double> v, vector<double> w) {
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->Or(v[i], w[i]);
	}

	return result;
}

vector<double> BulkLogic::Or2(vector<double> v, vector<double> w) {
	vector<double> result (v.size());

	for (int i = 0; i < v.size(); i++) {
		result[i] = this->fuzzy->Or2(v[i], w[i]);
	}

	return result;
}
				