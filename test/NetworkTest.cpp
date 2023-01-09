//
// Created by Robert Nickel on 1/8/23.
//
#include "gtest/gtest.h"
#include "../include/mathhelper.h"
#include "../include/network.h"


class NetworkTestSuite : public ::testing::Test {
protected:
    typedef std::vector<float> f_vec;
    virtual void SetUp() {
        networkSizes.push_back(std::pair<int,int>(2,Neuron::SIGMOID));
        networkSizes.push_back(std::pair<int,int>(2,Neuron::SIGMOID));
        networkSizes.push_back(std::pair<int,int>(1,Neuron::SIGMOID));
        testNetwork = new Network(networkSizes);

        // Make xor training data
        trainingInputs.push_back(0);
        trainingInputs.push_back(0);
        correctOuputs.push_back(0);
        trainingPairs.push_back(std::pair<f_vec,f_vec>(trainingInputs,correctOuputs));
        trainingInputs.clear();
        correctOuputs.clear();

        trainingInputs.push_back(0);
        trainingInputs.push_back(1);
        correctOuputs.push_back(1);
        trainingPairs.push_back(std::pair<f_vec,f_vec>(trainingInputs,correctOuputs));
        trainingInputs.clear();
        correctOuputs.clear();

        trainingInputs.push_back(1);
        trainingInputs.push_back(0);
        correctOuputs.push_back(1);
        trainingPairs.push_back(std::pair<f_vec,f_vec>(trainingInputs,correctOuputs));
        trainingInputs.clear();
        correctOuputs.clear();

        trainingInputs.push_back(1);
        trainingInputs.push_back(1);
        correctOuputs.push_back(0);
        trainingPairs.push_back(std::pair<f_vec,f_vec>(trainingInputs,correctOuputs));

        // Populate training pair vector
        for(int i = 0;i < 1000;i++){
            trainingPairs.push_back(trainingPairs[0]);
            trainingPairs.push_back(trainingPairs[1]);
            trainingPairs.push_back(trainingPairs[2]);
            trainingPairs.push_back(trainingPairs[3]);
/*            trainingPairs.insert(trainingPairs.end(),trainingPairs.begin(),trainingPairs.begin()+4);*/
        }
    }
    virtual void TearDown() {
        networkSizes.clear();
        delete testNetwork;
    }

    Network* testNetwork;
    std::vector<std::pair<int,int>> networkSizes;
    // For training once
    std::vector<float> trainingInputs;
    std::vector<float> correctOuputs;

    // For training multiple epochs
    std::vector<std::pair<f_vec,f_vec>> trainingPairs;
};

TEST_F(NetworkTestSuite, TestTrainingOnce) {
    for(std::pair<f_vec,f_vec> trainingPair:trainingPairs){
        testNetwork->train_once(trainingPair,0.02,0.3);
        std::cout << "Error: " << testNetwork->getError() << std::endl;
    }

    ASSERT_GE(2,1);
}