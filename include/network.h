#pragma once
#include <vector>
#include <map>
#include <string>
#include "neuron.h"

class Network {
public:

    Network(std::vector<std::pair<int,int>> &sizes);

    ~Network(){
        for(std::vector<Neuron*> layer:neuronLayers){
            for(Neuron* neuron:layer){
                delete neuron;
            }
        }
        neuronLayers.clear();
    }

    /*
     *  Functions
     */
    int train(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, float learningRate, float momentum, int epochs);
    std::vector<float> train_once(std::pair<std::vector<float>, std::vector<float>> &t_trainingData, float learningRate, float momentum);
    std::vector<float> predict(std::vector<float>& testData);
    int highestPred(std::vector<float> &outputNeurons);
    int save(std::string filename);
    int load(std::string filename);

    float getLearningRate() const;

    void setLearningRate(float learningRate);

    float getMomentum() const;

    void setMomentum(float momentum);

    int getEpochsToTrain() const;

    void setEpochsToTrain(int epochsToTrain);

    int getCurrentEpoch() const;

    void setCurrentEpoch(int currentEpoch);

    float getError() const {
        return m_error;
    }

    enum errorFunctions {
        SQUARED = 0,
    };
protected:
    void calcMSE(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, std::vector<std::vector<Neuron*>> &outputNeurons);
    void calcRMSE(std::pair<std::vector<float>, std::vector<float>> &trainingData, std::vector<Neuron*> &outputNeurons);
private:
    /*
     *  Functions
     */
    std::vector<Neuron*> feedForward(std::vector<float>& testData);
    
    float backProp(float deltaCurrent,float activationBefore);
    float backPropMomentum(float deltaCurrent, float activationBefore, float &oldChange);

    /*
     *  Class Variables
     */
    //Layer -> Neuron -> Weights (holds weights pointing into neuron)
/*    std::vector<std::vector<std::vector<float>>> weights;*/
    //Layer -> Neuron -> change

    //Layer -> Neuron, class values
    std::vector<std::vector<Neuron*>> neuronLayers;

    int epochsToTrain;
    int currentEpoch;
    int numLayers;
    float learningRate;
    float momentum;
    float m_error;
    unsigned int m_errorFunction;
    //Network sizes
    std::vector<std::pair<int,int>> sizes;

    void calculateError(Neuron *neuronToChange, float correctOuput);
};
