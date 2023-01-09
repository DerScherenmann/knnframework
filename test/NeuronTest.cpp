//
// Created by Robert Nickel on 1/8/23.
//
#include "gtest/gtest.h"
#include "../include/neuron.h"
#include "../include/mathhelper.h"


class NeuronTestSuite : public ::testing::Test {
protected:
    virtual void SetUp() {
        testNeuron = new Neuron(1,Neuron::NEURON,Neuron::SIGMOID);
        // Populate dummy weights
        for(int i = 0;i < 10;i++){
            previousWeights.push_back(i % 2);
        }
        for(int i = 0;i < 10;i++){
            previousNeurons.push_back(new Neuron(i % 2,Neuron::NEURON,Neuron::SIGMOID));
        }
    }
    virtual void TearDown() {
        delete testNeuron;
    }
    std::vector<Neuron*> previousNeurons;
    std::vector<float> previousWeights;

    Math math;

    Neuron* testNeuron;
};

/**
 * Test Neuron m_activation calculation
 */
TEST_F(NeuronTestSuite,TestCalculateActivation) {
    std::vector<Neuron*> oneNeuron(previousNeurons.begin(),previousNeurons.begin());
    testNeuron->calculateActivation(oneNeuron, previousWeights);

    // Activation should be sigmoid(0*1)
    ASSERT_FLOAT_EQ(math.sigmoid(0),testNeuron->getActivation());

    testNeuron->calculateActivation(previousNeurons,previousWeights);
    int testActivation = 0;
    for(int i = 0;i < previousNeurons.size();i++){
        testActivation += previousNeurons[i]->getActivation() * previousWeights[i];
    }
    ASSERT_FLOAT_EQ(math.sigmoid(testActivation),testNeuron->getActivation());
}

/**
 * Test if neuron skipped on bias
 */
TEST_F(NeuronTestSuite,TestSkipIfBias) {
    testNeuron->setType(Neuron::BIAS);
    testNeuron->setActivation(0.14151);
    testNeuron->calculateActivation(previousNeurons,previousWeights);
    ASSERT_FLOAT_EQ(0.14151,testNeuron->getActivation());
}