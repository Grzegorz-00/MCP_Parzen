#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

#include "KDE.h"
#include "Generator.h"


#define RESULT_SIZE 100
#define SAMPLE_SIZE 50
#define SAMPLE_QUANTITY 100

int main(int argc, char **argv)
{
	Data inputData(SAMPLE_SIZE, SAMPLE_QUANTITY);
	Data outputData(SAMPLE_SIZE, RESULT_SIZE);
	Generator::generateDoubleGauss(&inputData);

	KDE kde(&inputData, &outputData, -10, 10, 2, epanechnikov);
	kde.getResultCUDA();
	kde.saveResultToFile("result.txt");
	float val = kde.getChiSquaredVal();
	std::cout << val << std::endl;
	return 0;
}
