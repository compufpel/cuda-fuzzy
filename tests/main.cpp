#define CATCH_CONFIG_MAIN
#include "./Catch/single_include/catch.hpp"
#include "../lib/FuzzyNumber/FuzzyNumber.hpp"
#include "../lib/FuzzyNumber/Operation.hpp"
#include "../lib/FuzzyNumber/BulkOperation.hpp"
#include "../lib/FuzzyLogic/FuzzyLogic.hpp"
#include "../lib/FuzzyLogic/BulkLogic.hpp"
#include <iostream>
#include <vector>

using namespace std;

SCENARIO( "We can create FuzzyNumbers", "[fuzzy]" ) {

    GIVEN( "the fuzzy number library" ) {
        

        WHEN( "we create a new FuzzyNumber with begin=-1 and end=1 as parameters" ) {
			
		FuzzyNumber* fuzzyNumber = new FuzzyNumber(-1, 1);

            THEN( "the new object musst have these values as begin and end" ) {
                REQUIRE( fuzzyNumber->Begin() == -1 );
                REQUIRE( fuzzyNumber->End() == 1 );
            }
        }
        
    }
}

SCENARIO( "We can sum two Fuzzy Number", "[fuzzy]" ) {

    GIVEN( "2 Fuzzy Numbers" ) {
        
		FuzzyNumber* fuzzyNumber = new FuzzyNumber(0, 10);
		FuzzyNumber* fuzzyNumber2 = new FuzzyNumber(-4, 8);
		
        WHEN( "we can sum both" ) {
			
			FuzzyNumber* result = *fuzzyNumber + *fuzzyNumber2;
			

            THEN( "the result must be equals -4, 18" ) {
                REQUIRE( result->Begin() == -4 );
                REQUIRE( result->End() == 18 );
            }
        }
        
    }
}


SCENARIO( "We can subtract two Fuzzy Number", "[fuzzy]" ) {

    GIVEN( "2 Fuzzy Numbers" ) {
        
        FuzzyNumber* fuzzyNumber = new FuzzyNumber(0, 10);
        FuzzyNumber* fuzzyNumber2 = new FuzzyNumber(-4, 8);
        
        WHEN( "we can subtract both" ) {
            
            FuzzyNumber* result = *fuzzyNumber - *fuzzyNumber2;
            

            THEN( "the result must be equals -8, 14" ) {
                REQUIRE( result->Begin() == -8 );
                REQUIRE( result->End() == 14 );
            }
        }
        
    }           
}

SCENARIO( "We can multiply two Fuzzy Number", "[fuzzy]" ) {

    GIVEN( "2 Fuzzy Numbers" ) {
        
        FuzzyNumber* fuzzyNumber = new FuzzyNumber(2, 3);
        FuzzyNumber* fuzzyNumber2 = new FuzzyNumber(3, 4);
        
        WHEN( "we can multiply both" ) {
            
            FuzzyNumber* result = *fuzzyNumber * *fuzzyNumber2;
            

            THEN( "the result must be equals 12, 6" ) {
                REQUIRE( result->Begin() == 12 );
                REQUIRE( result->End() == 6 );
            }
        }
        
    }
}

SCENARIO( "We can division two Fuzzy Number", "[fuzzy]" ) {

    GIVEN( "2 Fuzzy Numbers" ) {
        
        FuzzyNumber* fuzzyNumber = new FuzzyNumber(24, 24);
        FuzzyNumber* fuzzyNumber2 = new FuzzyNumber(2, 4);
        
        WHEN( "we can division both" ) {
            
            FuzzyNumber* result = *fuzzyNumber / *fuzzyNumber2;
            

            THEN( "the result must be equals 12, 6" ) {
                REQUIRE( result->Begin() == 12 );
                REQUIRE( result->End() == 6 );
            }
        }
        
    }
}


SCENARIO( "We can inverse two Fuzzy Number", "[fuzzy]" ) {

    GIVEN( "2 Fuzzy Numbers" ) {
        
        FuzzyNumber* fuzzyNumber = new FuzzyNumber(1, 10);
        
        WHEN( "we can inverse" ) {
            
            FuzzyNumber* result = !(*fuzzyNumber);
            

            THEN( "the result must be equals 1, 0.1" ) {
                REQUIRE( result->Begin() == 0.1f );
                REQUIRE( result->End() == 1 );
            }
        }
        
    }
}

SCENARIO( "We can use operations", "[fuzzy]" ) {

    GIVEN( "4 operations" ) {
        
        FuzzyNumber* fuzzyNumber = new FuzzyNumber(1, 4);
        FuzzyNumber* fuzzyNumber2 = new FuzzyNumber(-2, 3);
		
		Operation* operation = new Operation(fuzzyNumber, fuzzyNumber2, '+');
		Operation* operation2 = new Operation(fuzzyNumber, fuzzyNumber2, '-');
		Operation* operation3 = new Operation(fuzzyNumber, fuzzyNumber2, '*');
		Operation* operation4 = new Operation(fuzzyNumber, fuzzyNumber2, '/');
        
        WHEN( "we execute the sum operation" ) {
            
            FuzzyNumber* result = operation->execute();
            
            THEN( "the result must be equals -1, 7" ) {
                REQUIRE( result->Begin() == -1 );
                REQUIRE( result->End() == 7 );
            }
			
			
        }
		
		WHEN( "we execute the minus operation" ) {
            
            FuzzyNumber* result = operation2->execute();
            
            THEN( "the result must be equals -2, 6" ) {
                REQUIRE( result->Begin() == -2 );
                REQUIRE( result->End() == 6 );
            }	
			
        }
		
		WHEN( "we execute the multiplication operation" ) {
            
            FuzzyNumber* result = operation3->execute();
            
            THEN( "the result must be equals 12, -2" ) {
                REQUIRE( result->Begin() == 12 );
                REQUIRE( result->End() == -2 );
            }	
			
        }
		
		WHEN( "we execute the division operation" ) {
            
            FuzzyNumber* result = operation4->execute();
            
            THEN( "the result must be equals -0.5" ) {
                REQUIRE( result->Begin() == -0.5 );
            }	
			
        }
		
    }
}

SCENARIO( "We can do a bulk of operations", "[fuzzy]" ) {

    GIVEN( "a bulk of operations" ) {
        
        FuzzyNumber* fuzzyNumber = new FuzzyNumber(1, 10);
        FuzzyNumber* fuzzyNumber2 = new FuzzyNumber(-2, 4);
		
		Operation* operation = new Operation(fuzzyNumber, fuzzyNumber2, '+');
		Operation* operation2 = new Operation(fuzzyNumber, fuzzyNumber2, '-');
		Operation* operation3 = new Operation(fuzzyNumber, fuzzyNumber2, '*');
		Operation* operation4 = new Operation(fuzzyNumber, fuzzyNumber2, '/');
		
		vector<Operation*> operations(4);
		
		operations[0] = operation;
		operations[1] = operation2;
		operations[2] = operation3;
		operations[3] = operation4;
		
		BulkOperation* bulk = new BulkOperation(operations);
        
        WHEN( "we can execute them all" ) {
            
			vector<FuzzyNumber*> result = bulk->execute();            

            THEN( "the result must be fine" ) {
				
				REQUIRE( result[0]->Begin() == -1 );
				REQUIRE( result[0]->End() == 14 );
				REQUIRE( result[1]->Begin() == -3 );
				REQUIRE( result[1]->End() == 12 );
				REQUIRE( result[2]->Begin() == 40 );
				REQUIRE( result[2]->End() == -2 );
				REQUIRE( result[3]->Begin() == -0.5 );
				REQUIRE( result[3]->End() == 2.5 );
				
            }
        }
        
    }
}

SCENARIO( "We can make a bulk of fuzzy logics", "[fuzzy]" ) {

    GIVEN( "a vector of doubles" ) {
        
        BulkLogic *bulk = new BulkLogic();
        vector<double> v (1000, 0.5);
        
        
        WHEN( "we use the not(!) operator on vector" ) {
            
            vector<double> result = bulk->Not(v);
            
            THEN( "the result should be 0.6" ) {
                REQUIRE( result[0] == 0.5 );
            }
        }
        
    }

    GIVEN( "2 vector of doubles" ) {
        
        BulkLogic *bulk = new BulkLogic();
        vector<double> v (1000, 0.3);
        vector<double> v2 (1000, 0.5);
        
        
        WHEN( "we use the and operator on both" ) {
            
            vector<double> result = bulk->And(v, v2);
            
            THEN( "the result should be 0.6" ) {
                REQUIRE( result[0] == 0.3 );
                REQUIRE( result[1] == 0.3 );
            }
        }
        
    }
}

