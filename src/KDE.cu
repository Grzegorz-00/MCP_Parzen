#include "KDE.h"

__device__ float epanechnikowKernelCUDA(float x)
{
	float res = 0;
	if(x <= 1 && x >= -1)
	{
		res = 3*(1-x*x)/4;
	}
	return res;
}

__device__ float uniformKernelCUDA(float x)
{
	float res = 0;
	if(x <= 1 && x >= -1)
	{
		res = 1.0/2;
	}
	return res;
}

__device__ float gaussianKernelCUDA(float x)
{
	float res = 0;
	res = 1.0/sqrt(2*M_PI)*exp(-1.0/2*x*x);
	return res;
}

__device__ float getSingleCUDA(float x, float h, int dataSize, float* data, int kernelType)
{
	float result = 0;
	for(int i = 0;i<dataSize;i++)
	{
		float kernel_par =(x-data[i])/h;
		if(kernelType == 0)
		{
			result += epanechnikowKernelCUDA(kernel_par);
		}
		else if(kernelType == 1)
		{
			result += uniformKernelCUDA(kernel_par);
		}
		else if(kernelType == 2)
		{
			result += gaussianKernelCUDA(kernel_par);
		}
	}
	result /= (dataSize*h);
	return result;
}

__global__ void getRangeCUDA(float* resultTable, int resultSize,float start, float step, float* data, int dataSize, float h, int kernelType)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < resultSize)
	{
		float val = start + idx*step;
		resultTable[idx] = getSingleCUDA(val,h,dataSize,data,kernelType);
	}
}

KDE::KDE(Data* inputData, Data* outputData, float start, float stop, float h, int kernelType)
{
	_kernelType = kernelType;
	_inputData = inputData;
	_outputData = outputData;
	_resultStart = start;
	_resultStop = stop;
	_h = h;
}

KDE::~KDE()
{

}

float KDE::epanechnikowKernel(float x)
{
	float res = 0;
	if(x <= 1 && x >= -1)
	{
		res = 3*(1-x*x)/4;
	}
	return res;
}

float KDE::uniformKernel(float x)
{
	float res = 0;
	if(x <= 1 && x >= -1)
	{
		res = 1.0/2;
	}
	return res;
}

float KDE::gaussianKernel(float x)
{
	float res = 0;
	res = 1.0/sqrt(2*M_PI)*exp(-1.0/2*x*x);
	return res;
}

float KDE::getSingle(float x, float* data)
{
	float res = 0;
	for(int i = 0;i<_inputData->getSampleSize();i++)
	{
		float kernel_par =(x-data[i])/_h;
		if(_kernelType == 0)
		{
			res += epanechnikowKernel(kernel_par);
		}
		else if(_kernelType == 1)
		{
			res += uniformKernel(kernel_par);
		}
		else if(_kernelType == 2)
		{
			res += gaussianKernel(kernel_par);
		}
	}
	res /= (_inputData->getSampleSize()*_h);
	return res;
}

void KDE::getResult()
{

	float step = (_resultStop-_resultStart)/(_outputData->getSampleSize()-1);

	for(int j = 0;j<_inputData->getSampleQuantity();j++)
	{
		float* tempInputData = _inputData->getDataPtr()[j];
		float* tempOutputData = _outputData->getDataPtr()[j];
		for(int i = 0;i<_outputData->getSampleSize();i++)
		{
			tempOutputData[i] = getSingle(_resultStart+step*i, tempInputData);
		}

		//normalize
		double sum = 0;
		for(int i = 0;i<_outputData->getSampleSize();i++)
		{
			sum += tempOutputData[i];
		}
		sum *= step;
		for(int i = 0;i<_outputData->getSampleSize();i++)
		{
			tempOutputData[i] /= sum;
		}

	}

	_outputData->compute();

}

void KDE::getResultCUDA()
{
	float step = (_resultStop-_resultStart)/(_outputData->getSampleSize()-1);

	int inputDataSize = _inputData->getSampleSize();
	int outputDataSize = _outputData->getSampleSize();

	int block_size = 512;
	int block_num = (outputDataSize + block_size - 1)/block_size;



	float *cudaDataTable;
	float *cudaResultTable;
	cudaMalloc((void**)&cudaDataTable,sizeof(float)*inputDataSize);
	cudaMalloc((void**)&cudaResultTable,sizeof(float)*outputDataSize);

	for(int j = 0;j<_inputData->getSampleQuantity();j++)
	{
		float* tempInputData = _inputData->getDataPtr()[j];
		float* tempOutputData = _outputData->getDataPtr()[j];

		cudaMemcpy(cudaDataTable,tempInputData,sizeof(float)*inputDataSize,cudaMemcpyHostToDevice);
		getRangeCUDA <<<block_num,block_size>>>(cudaResultTable, outputDataSize, _resultStart, step, cudaDataTable, inputDataSize, _h, _kernelType);
		cudaMemcpy(tempOutputData,cudaResultTable,sizeof(float)*outputDataSize,cudaMemcpyDeviceToHost);

		//normalize
		double sum = 0;
		for(int i = 0;i<_outputData->getSampleSize();i++)
		{
			sum += tempOutputData[i];
		}
		sum *= step;
		for(int i = 0;i<_outputData->getSampleSize();i++)
		{
			tempOutputData[i] /= sum;
		}
	}
	cudaFree(cudaDataTable);
	cudaFree(cudaResultTable);
	_outputData->compute();

}

void KDE::saveResultToFile(std::string filename)
{
	std::ofstream file;
	file.open(filename);
	float step = (_resultStop-_resultStart)/(_outputData->getSampleSize()-1);
	for(int i = 0;i<_outputData->getSampleSize();i++)
	{
		float mean = _outputData->getMean()[i];
		float dev = _outputData->getStdDev()[i];
		float x = _resultStart+step*i;
		float y = Generator::orginalDoubleGauss(x);
		file << x << " " << y << " " << mean-dev << " " << mean << " " << mean+dev << std::endl;
	}
	file.close();
}

float KDE::getChiSquaredVal()
{
	float step = (_resultStop-_resultStart)/(_outputData->getSampleSize()-1);
	double sum = 0;
	for(int i = 0;i<_outputData->getSampleSize();i++)
	{
		float mean = _outputData->getMean()[i];
		float dev = _outputData->getStdDev()[i];
		float x = _resultStart+step*i;
		float y = Generator::orginalDoubleGauss(x);
		double temp = (mean-y)/(dev);
		temp = temp*temp;
		sum += temp;
	}
	return sum;
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
