#include "BulkLogic.hpp"

BulkLogic::BulkLogic () {
	this->fuzzy = new FuzzyLogic();
}

vector<double> BulkLogic::Not(vector<double> v) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->Not(v[*it]);
	}

	return result;
}

vector<double> BulkLogic::Not2(vector<double> v) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->Not2(v[*it]);
	}

	return result;
}

vector<double> BulkLogic::Not3(vector<double> v) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->Not3(v[*it]);
	}

	return result;
}

vector<double> BulkLogic::And(vector<double> v, vector<double> w) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->And(v[*it], w[*it]);
	}

	return result;
}

vector<double> BulkLogic::And2(vector<double> v, vector<double> w) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->And2(v[*it], w[*it]);
	}

	return result;
}
			
vector<double> BulkLogic::Or(vector<double> v, vector<double> w) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->Or(v[*it], w[*it]);
	}

	return result;
}

vector<double> BulkLogic::Or2(vector<double> v, vector<double> w) {
	vector<double>::iterator it;
	vector<double> result (v.size());

	for (it = v.begin(); it != v.end(); ++it) {
		result[*it] = this->fuzzy->Or2(v[*it], w[*it]);
	}

	return result;
}
				