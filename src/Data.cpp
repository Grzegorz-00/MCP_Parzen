#include "Data.h"

Data::Data(int sampleSize, int sampleQuantity)
{
	_sampleSize = sampleSize;
	_sampleQuantity = sampleQuantity;
	_means = new float[_sampleSize];
	_stdDeviations = new float[_sampleSize];
	_data = new float*[_sampleQuantity];
	for(int i = 0;i<_sampleQuantity;i++)
	{
		_data[i] = new float[sampleSize];
	}
}
Data::~Data()
{
	for(int i = 0;i<_sampleQuantity;i++)
	{
		delete[] _data[i];
	}
	delete[] _data;
	delete[] _means;
	delete[] _stdDeviations;
}

void Data::compute()
{
	for(int i = 0;i<_sampleSize;i++)
	{
		double mean = 0;
		for(int j = 0;j<_sampleQuantity;j++)
		{
			mean += _data[j][i];
		}
		mean = mean/_sampleQuantity;
		_means[i] = mean;

		double dev = 0;
		for(int j = 0;j<_sampleQuantity;j++)
		{
			dev += (_data[j][i] - mean)*(_data[j][i] - mean);
		}
		dev = sqrt(dev/(_sampleQuantity-1));
		_stdDeviations[i] = dev;
	}
}

int Data::getSampleSize()
{
	return _sampleSize;
}
int Data::getSampleQuantity()
{
	return _sampleQuantity;
}
float** Data::getDataPtr()
{
	return _data;
}

float* Data::getMean()
{
	return _means;
}

float* Data::getStdDev()
{
	return _stdDeviations;
}
