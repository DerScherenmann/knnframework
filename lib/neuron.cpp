#include <vector>
#include <iostream>
#include "../include/neuron.h"

//neurons holds neurons of the layer before and therefore their m_activation, weights are the weights to this neuron
void Neuron::calculateActivation(std::vector<Neuron*> &neurons) {
    //skip this if neuron is bias
    if (m_neuronType == NEURON) {
        m_activation = 0;
        for (int i = 0; i < neurons.size(); i++) {
            m_activation += neurons[i]->getActivation() * m_weights[i];
        }
        setSum(m_activation);
        setActivation(act(m_activation));
    }
}

float Neuron::act(float x){

	if(m_activationFunction == SWISH){
		x = math.swish(x);
	}
	if(m_activationFunction == SIGMOID){
		x = math.sigmoid(x);
	}
	return x;
}

float Neuron::actPrime(float x){

	if(m_activationFunction == SWISH){
		x = math.swishPrime(x);
	}
	if(m_activationFunction == SIGMOID){
		x = math.sigmoidPrime(x);
	}

	return x;
}
