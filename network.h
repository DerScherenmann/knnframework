#pragma once
#include <vector>
#include <map>
#include <string>
#include "neuron.h"

class Network
{
public:

    /*
     * Constructor
     */
    /**
    * @param layersize layermfunction
    */
    Network(std::vector<std::pair<int,int>> sizes);

    /*
     *  Functions
     */
    int train(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, float learningRate, float momentum, int epochs);
    std::vector<float> train_once(std::pair<std::vector<float>, std::vector<float>> &pair,float learningRate,float momentum);
    std::vector<float> predict(std::vector<float>& testData);
    int highestPred(std::vector<float> &outputNeurons);
    int save(std::string filename);
    int load(std::string filename);

    /*
     *  Variables
     */
    int numLayers;
    float learningRate;
    float momentum;
    //Network sizes
    std::vector<std::pair<int,int>> sizes;

private:
    /*
     *  Functions
     */
    std::vector<Neuron> feedForward(std::vector<float>& testData);
    
    float backProp(float &deltaCurrent,float &activationBefore);
    float backPropMomentum(float &deltaCurrent, float &activationBefore, float &oldChange);

    float calcMSE(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, std::vector<std::vector<Neuron>> &outputNeurons);
    float calcRMSE(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, std::vector<std::vector<Neuron>> &outputNeurons);

    /*
     *  Variables
     */
    //Layer -> Neuron -> Weights (holds weights pointing into neuron)
    std::vector<std::vector<std::vector<float>>> weights;
    //Layer -> Neuron -> change
    std::vector<std::vector<std::vector<float>>> oldchange;
    //Layer -> Neuron -> class values
    std::vector<std::vector<Neuron>> neuronLayers;
};
