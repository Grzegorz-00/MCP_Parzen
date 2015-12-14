
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include "KDE.h"
#include <time.h>

#define SIZE 100000

using namespace std;



int main(int argc, char **argv)
{
	srand(time(NULL));
	float data[SIZE];
	for(int i = 0;i<SIZE;i++)
	{
		data[i] = rand()%4000-2000;
	}
	KDE kde1(SIZE,200,data);

	FILE* file = fopen("data.out","w");
	if(file == NULL)return 1;

	clock_t c_start,c_stop;

	const float start = -2200;
	const float stop = 2200;
	const float step = 0.5;
	int dataSize = (stop-start)/step;
	float *result = new float[dataSize];
	float *result2 = new float[dataSize];

	c_start = clock();
	kde1.getResult(start, step, dataSize, result);
	c_stop = clock();

	float cpu_time = (float)(c_stop-c_start)/(CLOCKS_PER_SEC/1000);
	printf("CPU execution time: %f ms\r\n",cpu_time);

	c_start = clock();
	kde1.getResultCUDA(start, step, dataSize, result2);
	c_stop = clock();

	float gpu_time = (float)(c_stop-c_start)/(CLOCKS_PER_SEC/1000);
	printf("CUDA execution time: %f ms\r\n",gpu_time);

	printf("CPU:CUDA %fx\r\n",cpu_time/gpu_time);

	for(int i = 0;i<dataSize;i++)
	{
		fprintf(file,"%f %f %f\r\n",start + step*i,result[i],result2[i]);
		//printf("%f %f %f\r\n",start + step*i,result[i],result2[i]);
	}

	//printf("%f \r\n",kde1.getSingle(0));

	delete[] result;
	delete[] result2;
	fclose(file);

	return 0;
}
