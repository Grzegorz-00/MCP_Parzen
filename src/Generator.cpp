#include "Generator.h"

float Generator::getGauss(float x, float mean, float variance)
{
	float y = 1.0/(variance*sqrt(2*M_PI))*exp(-(x-mean)*(x-mean)/(2*variance*variance));
	return y;
}

void Generator::generateDoubleGauss(Data* data)
{
	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<float> distribution1(5,2.5);
	std::normal_distribution<float> distribution2(-5,2.5);
	std::uniform_int_distribution<int> distribution0(0,1);

	float** rawData = data->getDataPtr();

	for(int i = 0;i<data->getSampleQuantity();i++)
	{
		for(int j = 0;j<data->getSampleSize();j++)
		{

			if(distribution0(generator))
			{
				rawData[i][j] = distribution1(generator);
			}
			else
			{
				rawData[i][j] = distribution2(generator);
			}

		}
	}
}


float Generator::orginalDoubleGauss(float x)
{
	float y = Generator::getGauss(x,-5,2.5)/2+Generator::getGauss(x,5,2.5)/2;
	return y;
}


void Generator::generateTripleGauss(Data* data)
{
	std::default_random_engine generator(std::random_device{}());
	std::uniform_int_distribution<int> distribution0(0,4);
	std::normal_distribution<float> distribution1(-4.5,1);
	std::normal_distribution<float> distribution2(0,1);
	std::normal_distribution<float> distribution3(4.5,1);

	float** rawData = data->getDataPtr();

	for(int i = 0;i<data->getSampleQuantity();i++)
	{
		for(int j = 0;j<data->getSampleSize();j++)
		{
			int rand = distribution0(generator);
			if(rand==0)
			{
				rawData[i][j] = distribution1(generator);
			}
			else if(rand==2)
			{
				rawData[i][j] = distribution3(generator);
			}
			else
			{
				rawData[i][j] = distribution2(generator);
			}
		}
	}
}

float Generator::orginalTripleGauss(float x)
{
	float y = Generator::getGauss(x,-4.5,1)+Generator::getGauss(x,0,1)*3+Generator::getGauss(x,4.5,1);
	return y;
}


