#ifndef NEURON_H
#define NEURON_H
#pragma once
#include <vector>

#include "mathhelper.h"


class Neuron {
	public:
        /**
         * Neuron Constructor which creates one neuron to include in network
         *
         * @param activation between 0-1 m_activation of this neuron
         * @param type enum Neuron m_neuronType
         * @param activationFunction enum m_activation function for this neuron
         */
        Neuron(float activation,int type,int activationFunction) : m_activation(activation), m_neuronType(type), m_activationFunction(activationFunction){
            if (type == BIAS) activation = 1;
        };
        ~Neuron(){
        }
		enum NeuronType {
			BIAS = 0,NEURON
		};
		enum NeuronMode {
			SWISH = 0,SIGMOID,RELU
		};
		void calculateActivation(std::vector<Neuron*> &neurons, std::vector<float> &weights);
		float act(float x);
		float actPrime(float x);

    float getActivation() const {
        return m_activation;
    }

    void setActivation(float activation) {
        Neuron::m_activation = activation;
    }

    float getSum() const {
        return m_sum;
    }

    void setSum(float sum) {
        Neuron::m_sum = sum;
    }

    float getDelta() const {
        return m_delta;
    }

    void setDelta(float delta) {
        Neuron::m_delta = delta;
    }

    int getType() const {
        return m_neuronType;
    }

    void setType(int type) {
        Neuron::m_neuronType = type;
    }

    int getMode() const {
        return m_activationFunction;
    }

    void setMode(int mode) {
        Neuron::m_activationFunction = mode;
    }

private:
        Math math;
        float m_activation = 0;
        /**
         * Sum is the cummulated m_activation of all previous neuron before it is multiplied with any function
         */
        float m_sum = 0;
        float m_delta = 0;
        int m_neuronType = 1;
        int m_activationFunction = 0;
};
#endif
