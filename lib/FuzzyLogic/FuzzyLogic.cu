/* CudaFuzzy Project - 2015 
 * Graduate Program in Computer Science - UFPel
 *
 * \file FuzzyLogic.cu
 *      This file contains the sequential implementations of fuzzy functions applied to arrays.
 */

#include "FuzzyLogic.cuh"

/*___Not___
*   Function: Implement the sequential version of Fuzzy not operation (1-Input).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double Not(double x) {
	return 1 - x;
}

/*___Not2___
*   Function: Implement the sequential version of Fuzzy not operation sqrt(1 - pow(x, 2)).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double Not2(double x) {
	return sqrt(1 - pow(x, 2));
}

/*___Not3___
*   Function: Implement the sequential version of Fuzzy not operation pow(1 - pow(x, 3), 1.0 / 3).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double Not3(double x) {
	return pow(1 - pow(x, 3), 1.0 / 3);
}

/*___And___
*   Function: Implement the sequential version of Fuzzy and operation (<).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Input:  double* y: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double And(double x, double y) {
	return x < y ? x : y;
}

/*___And2___
*   Function: Implement the sequential version of Fuzzy and operation (*).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Input:  double* y: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double And2(double x, double y) {
	return x * y;
}

/*___Or___
*   Function: Implement the sequential version of Fuzzy not operation (>).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Input:  double* y: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double Or(double x, double y) {
	return x > y ? x : y;
}

/*___Or2___
*   Function: Implement the sequential version of Fuzzy not operation (( x + y ) - ( x * y )).
*   Parameters:
*   Input:  double* x: Input of Basic Fuzzy operation
*   Input:  double* y: Input of Basic Fuzzy operation
*   Output: double   : Response of Fuzzy operation 
*   Creation date: November, 2015.
*   Exception case: -
*/
double Or2(double x, double y) {
	return ( x + y ) - ( x * y );
}


