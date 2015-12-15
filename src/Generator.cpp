#include "Generator.h"

void Generator::generateDoubleGauss(float* data, int dataSize)
{
	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<float> distribution1(5,2.5);
	std::normal_distribution<float> distribution2(-5,2.5);
	std::uniform_int_distribution<int> distribution0(0,1);

	for(int i = 0;i<dataSize;i++)
	{
		if(distribution0(generator))
		{
			data[i] = distribution1(generator);
		}
		else
		{
			data[i] = distribution2(generator);
		}
	}
}


void Generator::generateTripleGauss(float* data, int dataSize)
{
	std::default_random_engine generator(std::random_device{}());
	std::uniform_int_distribution<int> distribution0(0,4);
	std::normal_distribution<float> distribution1(-4.5,1);
	std::normal_distribution<float> distribution2(0,1);
	std::normal_distribution<float> distribution3(4.5,1);


	for(int i = 0;i<dataSize;i++)
	{
		int rand = distribution0(generator);
		if(rand==0)
		{
			data[i] = distribution1(generator);
		}
		else if(rand==2)
		{
			data[i] = distribution3(generator);
		}
		else
		{
			data[i] = distribution2(generator);
		}
	}
}


