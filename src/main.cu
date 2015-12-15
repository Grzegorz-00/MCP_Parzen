
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "KDE.h"
#include "Generator.h"

#define SAMPLE_SIZE 4000
#define RESULT_SIZE 1000

int main(int argc, char **argv)
{

	clock_t start, stop;
	float time;

	float *inputData = new float[SAMPLE_SIZE];

	Generator::generateTripleGauss(inputData, SAMPLE_SIZE);
	KDE::saveHistToFile("testHist.txt",inputData,SAMPLE_SIZE,30);
	KDE kernel(SAMPLE_SIZE,2,inputData);
	start = clock();
	kernel.getResult(-15,15,RESULT_SIZE);
	stop = clock();
	time = (float)(stop-start)/CLOCKS_PER_SEC;
	std::cout << "CPU time: " << time << std::endl;

	start = clock();
	kernel.getResultCUDA(-15,15,RESULT_SIZE);
	stop = clock();
	time = (float)(stop-start)/CLOCKS_PER_SEC;
	std::cout << "CUDA time: " << time << std::endl;



	kernel.saveResultToFile("output.txt");

	delete[] inputData;
	return 0;
}
