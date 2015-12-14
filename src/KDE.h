#ifndef KDE_H_
#define KDE_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

class KDE
{
private:
	int _dataSize;
	float *_data;
	float *_args;
	float _h;
	float epanechnikowKernel(float x);
	//CUDA
	float *cuda_data;
	float *cuda_args;


public:
	KDE(int size, float h, float* data);
	~KDE();
	float getSingle(float x);
	void getResult(float start, float step, int size, float* result);
	void getResultCUDA(float start, float step, int size, float* result);


};



#endif
