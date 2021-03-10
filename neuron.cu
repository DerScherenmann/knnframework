#include <vector>
#include <iostream>
#include "neuron.h"

//activation is a number held by the Neuron
Neuron::Neuron(float activation,int type,int mode) {
	this->activation = activation;
	this->type = type;
	this->mode = mode;
	//set activation to 1 if bias
	if (type == BIAS) activation = 1;
}

//neurons holds neurons of the layer before and therefore their activation, weights are the weights to this neuron
void Neuron::calculateActivation(std::vector<Neuron> &neurons, std::vector<float> &weights) {
    //skip this if neuron is bias

    if (type == NEURON) {
        activation = 0;

        for (int i = 0; i < neurons.size(); i++) {
            activation += neurons[i].activation * weights[i];
        }

        this->sum = activation;
        this->activation = act(activation);
        //std::cout << "Activation: " << activation << std::endl;
    }
}

float Neuron::act(float x){

	if(mode == SWISH){
		x = math.swish(x);
	}
	if(mode == SIGMOID){
		x = math.sigmoid(x);
	}
	return x;
}

float Neuron::actPrime(float x){

	if(mode == SWISH){
		x = math.swishPrime(x);
	}
	if(mode == SIGMOID){
		x = math.sigmoidPrime(x);
	}

	return x;
}
