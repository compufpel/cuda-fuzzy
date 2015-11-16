test:
	nvcc lib/FuzzyNumber/*.cpp lib/FuzzyLogic/*.cu tests/main.cu  -o tests/teste
	./tests/teste
	


