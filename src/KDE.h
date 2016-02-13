#ifndef KDE_H_
#define KDE_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "Data.h"
#include "Generator.h"

enum kernel_type
{
	uniform = 0,
	epanechnikov = 1,
	gaussian = 2
};

class KDE
{
private:
	Data* _inputData;
	Data* _outputData;
	float _resultStart;
	float _resultStop;
	float _h;
	bool _errorOccur = false;
	kernel_type _kernelType;

	float epanechnikowKernel(float x);
	float uniformKernel(float x);
	float gaussianKernel(float x);

	void notifyCudaAllocError();
	void notifyCudaCpyError();


public:
	KDE(Data* inputData, Data* outputData, float start, float stop, float h, kernel_type kernelType);
	~KDE();
	float getSingle(float x, float* data);
	void getResult();
	void getResultCUDA();
	void saveResultToFile(std::string filename);
	float getChiSquaredVal();
	static void saveHistToFile(std::string filename, float* data, int dataSize, int bids);
};



#endif
