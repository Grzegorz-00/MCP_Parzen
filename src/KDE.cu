#include "KDE.h"

__device__ float epanechnikowKernelCUDA(float x)
{
	float res = 0;
	if(x <= 1 && x >= -1)
	{
		res = 3*(1-x*x);
	}
	return res;
}

__device__ float getSingleCUDA(float x, float h, int dataSize, float* data)
{
	float result = 0;
	for(int i = 0;i<dataSize;i++)
	{
		float kernel_par =(x-data[i])/h;
		result += epanechnikowKernelCUDA(kernel_par);
	}
	result /= (dataSize*h);
	return result;
}

__global__ void getRangeCUDA(float* args, int size, float* data, float h, int dataSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		args[idx] = getSingleCUDA(args[idx],h,dataSize,data);
	}
}

KDE::KDE(int size, float h, float* data)
{
	if(data != NULL)
	{
		_dataSize = size;
		_h = h;
		int block_size = 512;
		int block_num = (_dataSize + block_size - 1)/block_size;
		_data = new float[size];
		for(int i = 0;i<size;i++)
		{
			_data[i] = data[i];
		}

		//CUDA
		cudaMalloc((void **)&cuda_data,sizeof(float)*_dataSize);
		cudaMemcpy(cuda_data,_data,sizeof(float)*_dataSize,cudaMemcpyHostToDevice);
	}
}

KDE::~KDE()
{
	delete[] _data;
	_dataSize = 0;
	cudaFree(cuda_data);
}

float KDE::epanechnikowKernel(float x)
{
	float res = 0;
	if(x <= 1 && x >= -1)
	{
		res = 3*(1-x*x);
	}
	return res;
}
float KDE::getSingle(float x)
{
	float res = 0;
	for(int i = 0;i<_dataSize;i++)
	{
		float kernel_par =(x-_data[i])/_h;
		res += epanechnikowKernel(kernel_par);
	}
	res /= (_dataSize*_h);
	return res;
}

void KDE::getResult(float start, float step, int size, float* result)
{
	for(int i = 0;i<size;i++)
	{
		result[i] = getSingle(start+step*i);
	}
}


void KDE::getResultCUDA(float start, float step, int size, float* result)
{
	int block_size = 512;
	int block_num = (size + block_size - 1)/block_size;

	for(int i = 0;i<size;i++)
	{
		result[i] = start+step*i;
	}

	cudaMalloc((void**)&cuda_args,sizeof(float)*size);
	cudaMemcpy(cuda_args,result,sizeof(float)*size,cudaMemcpyHostToDevice);

	getRangeCUDA <<<block_num,block_size>>>(cuda_args, size, cuda_data, _h, _dataSize);

	cudaMemcpy(result,cuda_args,sizeof(float)*size,cudaMemcpyDeviceToHost);
	cudaFree(cuda_args);
}
