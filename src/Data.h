#ifndef DATA_H_
#define DATA_H_

#include <cmath>
#include <iostream>

class Data
{
private:
	float** _data;
	float* _means;
	float* _stdDeviations;
	int _sampleSize;
	int _sampleQuantity;

public:
	Data(int sampleSize, int sampleQuantity);
	~Data();
	void compute();
	void print();
	int getSampleSize();
	int getSampleQuantity();
	float** getDataPtr();
	float* getMean();
	float* getStdDev();
};



#endif
