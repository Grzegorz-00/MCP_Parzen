#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>

#include "Data.h"
#include <cmath>

class Generator
{
private:
	static float getGauss(float x, float mean, float variance);
public:
	static void generateDoubleGauss(Data* data);
	static void generateTripleGauss(Data* data);
	static float orginalDoubleGauss(float x);
	static float orginalTripleGauss(float x);
};





#endif
