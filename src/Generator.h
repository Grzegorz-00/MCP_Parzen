#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <cstdlib>
#include <ctime>
#include <random>

class Generator
{
public:
	static void generateDoubleGauss(float *data, int dataSize);
	static void generateTripleGauss(float *data, int dataSize);
};





#endif
