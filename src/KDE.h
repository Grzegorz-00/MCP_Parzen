#ifndef KDE_H_
#define KDE_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

class KDE
{
private:
	int _dataSize;
	int _resultSize;
	float _resultStart;
	float _resultStop;
	float *_data;
	float *_resultData;
	float _h;
	bool _errorOccur = false;

	float epanechnikowKernel(float x);
	void notifyCudaAllocError();
	void notifyCudaCpyError();


public:
	KDE(int size, float h, float* data);
	~KDE();
	float getSingle(float x);
	void getResult(float start, float stop, int resultSize);
	void getResultCUDA(float start, float stop, int resultSize);
	void saveResultToFile(std::string filename);
	static void saveHistToFile(std::string filename, float* data, int dataSize, int bids);
};



#endif
