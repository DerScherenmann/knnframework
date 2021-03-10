#ifndef NEURON_H
#define NEURON_H
#pragma once
#include <vector>

#include "mathhelper.h"


class Neuron
{
	public:
        Neuron(float activation,int type,int mode);

		float activation = 0;
		float sum = 0;
		float delta = 0;
		int type = 1;
		int bias = 1;
		enum types
		{
			BIAS = 0,NEURON
		};
		enum modes{
			SWISH = 0,SIGMOID,RELU
		};
		void calculateActivation(std::vector<Neuron> &neurons, std::vector<float> &weights);
		float act(float x);
		float actPrime(float x);
		int mode = 0;
    private:
        Math math;
};
#endif
