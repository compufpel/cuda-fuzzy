#define CATCH_CONFIG_MAIN
#include "./Catch/single_include/catch.hpp"
#include "../lib/FuzzyNumber.hpp"

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