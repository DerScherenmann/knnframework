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
        networkSizes.emplace_back(2,Neuron::SIGMOID);
        networkSizes.emplace_back(2,Neuron::SIGMOID);
        networkSizes.emplace_back(1,Neuron::SIGMOID);
        testNetwork = new Network(networkSizes);

        // Make xor training data
        trainingInputs.push_back(0);
        trainingInputs.push_back(0);
        correctOuputs.push_back(0);
        trainingPairs.emplace_back(trainingInputs,correctOuputs);
        trainingInputs.clear();
        correctOuputs.clear();

        trainingInputs.push_back(0);
        trainingInputs.push_back(1);
        correctOuputs.push_back(1);
        trainingPairs.emplace_back(trainingInputs,correctOuputs);
        trainingInputs.clear();
        correctOuputs.clear();

        trainingInputs.push_back(1);
        trainingInputs.push_back(0);
        correctOuputs.push_back(1);
        trainingPairs.emplace_back(trainingInputs,correctOuputs);
        trainingInputs.clear();
        correctOuputs.clear();

        trainingInputs.push_back(1);
        trainingInputs.push_back(1);
        correctOuputs.push_back(0);
        trainingPairs.emplace_back(trainingInputs,correctOuputs);

        // Populate training pair vector
        for(int i = 0;i < 10000;i++){
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

TEST_F(NetworkTestSuite, TestTraining) {
    std::cout << "Testing Network::train()" << std::endl;

    float learningRate = 0.4;
    float momentum = 0.7;
    testNetwork->train(trainingPairs,learningRate,momentum,100);

    ASSERT_GE(0.3,testNetwork->getError());
}

TEST_F(NetworkTestSuite, TestTrainingOnce) {
    std::cout << "Testing Network::train_once()" << std::endl;

    // This should do the same as Network::train but with a different function used as api for my Convolutional Network Project
    // this emulates epochs
    float lastError = 0;
    float learningRate = 0.05;
    float momentum = 0.3;
    for(int i = 0;i < 5000;i++){
        // For all training images
        for(std::pair<f_vec,f_vec> &trainingPair:trainingPairs){
            testNetwork->train_once(trainingPair,learningRate,momentum);
            if(testNetwork->getError() >= lastError){
                if(momentum < 1){
                    momentum += 0.05;
                }
                if(learningRate < 1){
                    learningRate += 0.05;
                }
            }
            if(testNetwork->getError() < lastError+learningRate){
                if(momentum > 0.3){
                    momentum -= 0.05;
                }
                if(learningRate > 0.05){
                    learningRate -= 0.05;
                }
            }
        }
        lastError = testNetwork->getError();
        // For debug purposes
        std::cout << "Error: " << lastError << std::endl;
    }

    ASSERT_GE(0.3,testNetwork->getError());
}

TEST_F(NetworkTestSuite, TestFeedForward) {
    // Set threshold for test
    float threshold = 0.3;
    // Run feed forward test for all training pairs
    for(std::pair<f_vec,f_vec> &trainingPair:trainingPairs){
        f_vec predictions = testNetwork->predict(trainingPair.first);
        // Only one ouput neuron so we can use [0], xor network
        if(trainingPair.second[0] == 0.0f){
            std::cout << trainingPair.first[0] << " " << trainingPair.first[1] << " -> " << predictions[0] << " < " << trainingPair.second[0]+threshold << std::endl;
            GTEST_ASSERT_LT(predictions[0],trainingPair.second[0]+threshold);
        }else{
            std::cout << trainingPair.first[0] << " " << trainingPair.first[1] << " -> " << predictions[0] << " > " << trainingPair.second[0]-threshold << std::endl;
            GTEST_ASSERT_GT(predictions[0],trainingPair.second[0]-threshold);
        }
    }
}