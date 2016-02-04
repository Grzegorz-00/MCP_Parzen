
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "KDE.h"
#include "Generator.h"

#define SAMPLE_SIZE 100
#define RESULT_SIZE 100
#define SAMPLE_QUANTITY 100

int main(int argc, char **argv)
{


	Data inputData(SAMPLE_SIZE, SAMPLE_QUANTITY);
	Data outputData1(SAMPLE_SIZE, RESULT_SIZE);
	Data outputData2(SAMPLE_SIZE, RESULT_SIZE);
	Data outputData3(SAMPLE_SIZE, RESULT_SIZE);
	Data outputData4(SAMPLE_SIZE, RESULT_SIZE);
	Data outputData5(SAMPLE_SIZE, RESULT_SIZE);

	Generator::generateDoubleGauss(&inputData);

	KDE kde1(&inputData, &outputData1, -10, 10, 0.1, 0);
	KDE kde2(&inputData, &outputData2, -10, 10, 0.5, 0);
	KDE kde3(&inputData, &outputData3, -10, 10, 1, 0);
	KDE kde4(&inputData, &outputData4, -10, 10, 2, 0);
	KDE kde5(&inputData, &outputData5, -10, 10, 5, 0);
	kde1.getResult();
	kde2.getResult();
	kde3.getResult();
	kde4.getResult();
	kde5.getResult();
	kde1.saveResultToFile("res3a.txt");
	kde2.saveResultToFile("res3b.txt");
	kde3.saveResultToFile("res3c.txt");
	kde3.saveResultToFile("res3d.txt");
	kde3.saveResultToFile("res3e.txt");

	return 0;
}
