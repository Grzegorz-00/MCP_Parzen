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

__global__ void getRangeCUDA(float* resultTable, int resultSize, float start, float stop, float* data, int dataSize, float h)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < resultSize)
	{
		float val = start + idx*(stop-start)/(resultSize-1);
		resultTable[idx] = getSingleCUDA(val,h,dataSize,data);
	}
}

KDE::KDE(int size, float h, float* data)
{
	_resultData = NULL;
	_resultSize = 0;
	if(data != NULL)
	{
		_dataSize = size;
		_h = h;
		_data = new float[size];
		for(int i = 0;i<size;i++)
		{
			_data[i] = data[i];
		}
	}
}

KDE::~KDE()
{
	delete[] _data;
	_dataSize = 0;
	if(_resultData != NULL)
	{
		delete[] _resultData;
		_resultSize = 0;
	}
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

void KDE::getResult(float start, float stop, int resultSize)
{
	_resultSize = resultSize;
	_resultStart = start;
	_resultStop = stop;

	if(_resultData != NULL)
	{
		delete[] _resultData;
	}
	_resultData = new float[_resultSize];
	float step = (_resultStop-_resultStart)/(_resultSize-1);

	for(int i = 0;i<_resultSize;i++)
	{
		_resultData[i] = getSingle(_resultStart+step*i);
	}
}

void KDE::saveResultToFile(std::string filename)
{
	std::ofstream file;
	file.open(filename);
	float step = (_resultStop-_resultStart)/(_resultSize-1);
	for(int i = 0;i<_resultSize;i++)
	{
		file <<  (_resultStart+step*i) << " " << getSingle(_resultStart+step*i) << std::endl;
	}
}


void KDE::getResultCUDA(float start, float stop, int resultSize)
{
	if(_errorOccur)
	{
		std::cout << "Poprzednia operacja zakończona niepowodzeniem" << std::endl;
		return;
	}
	_resultSize = resultSize;
	_resultStart = start;
	_resultStop = stop;

	if(_resultData != NULL)
	{
		delete[] _resultData;
	}
	_resultData = new float[_resultSize];
	int block_size = 512;
	int block_num = (resultSize + block_size - 1)/block_size;


	float *cudaDataTable;
	float *cudaResultTable;

	if(cudaMalloc((void**)&cudaDataTable,sizeof(float)*_dataSize)!=cudaSuccess)notifyCudaAllocError();
	if(cudaMalloc((void**)&cudaResultTable,sizeof(float)*_resultSize)!=cudaSuccess)notifyCudaAllocError();
	if(!_errorOccur)
	{
		if(cudaMemcpy(cudaDataTable,_data,sizeof(float)*_dataSize,cudaMemcpyHostToDevice)!=cudaSuccess)notifyCudaCpyError();
	}
	if(!_errorOccur)
	{
		getRangeCUDA <<<block_num,block_size>>>(cudaResultTable, _resultSize, _resultStart, _resultStop, cudaDataTable, _dataSize, _h);
		cudaMemcpy(_resultData,cudaResultTable,sizeof(float)*_resultSize,cudaMemcpyDeviceToHost);
	}

	cudaFree(cudaDataTable);
	cudaFree(cudaResultTable);
}

void KDE::saveHistToFile(std::string filename, float* data, int dataSize, int bids)
{
	std::ofstream file;
	file.open(filename);
	float min = data[0];
	float max = data[0];
	for(int i = 1;i<dataSize;i++)
	{
		if(data[i] > max)max = data[i];
		else if(data[i] < min)min = data[i];
	}
	double stepSize = (double)(max-min)/bids;

	for(int i = 0;i<bids-1;i++)
	{
		double minRange = min + i*stepSize;
		double maxRange = min + (i+1)*stepSize;
		int rangeOccurs = 0;
		for(int j = 0;j<dataSize;j++)
		{
			if(data[j] >= minRange && data[j] < maxRange)rangeOccurs++;
		}

		file << (minRange+maxRange)/2 << " " << rangeOccurs << std::endl;
	}
	file.close();
}

void KDE::notifyCudaAllocError()
{
	std::cout << "CUDA Alloc problem" << std::endl;
	_errorOccur = true;
}

void KDE::notifyCudaCpyError()
{
	std::cout << "CUDA Memcpy problem" << std::endl;
	_errorOccur = true;
}
