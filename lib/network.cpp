#include <algorithm>
#include "../include/mathhelper.h"
#include "../include/network.h"
#include "../include/neuron.h"
#include <map>
#include <iostream>
#include <future>
#include <string>
#include <fstream>
#include <utility>

#include <chrono>
#include <csignal>

Math math;

int averageTimeChange = 0;
int timesChange = 0;
int averageTimeBackprop = 0;
int timesBackprop = 0;
int averageTimeEpoch = 0;
int timesEpoch = 100;
int averageTimeForward = 0;
int timesForward = 100;

volatile int signalFlag;
// TODO maybe there is a better way
/**
 * Handles SIGINT and stops training
 */
void static_handleSignalInterrupt(int s){
    signalFlag = s;
}

/**
 * Constructs a neural network
 * @param sizes pair which holds layer size and m_activationFunction
 */
Network::Network(std::vector<std::pair<int,int>> &sizes) {
    // Initialize signal catcher
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = static_handleSignalInterrupt;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

	this->numLayers = sizes.size();
	this->sizes = sizes;

	int y = 1;
	for (int x = 0; x < numLayers;x++) {
		if (y >= numLayers){
            break;
		}
		//inititalize weight vector with random numbers
		weights.push_back(math.defmatrix(sizes[x].first+1, sizes[y].first));
		//initialize gradient vector with random stuff
		oldchange.push_back(math.defmatrix(sizes[x].first+1, sizes[y].first));
		y++;
	}
    
	//zero old changes
	for (int i = 0; i < oldchange.size(); i++) {
		for (int j = 0; j < oldchange[i].size(); j++) {
			for (int k = 0; k < oldchange[i][j].size(); k++) {
				oldchange[i][j][k] = 0;
			}
		}
	}

	std::vector<Neuron*> neurons;

	//define our input neuron(s)
	for (int j = 0; j < sizes[0].first; j++) {
		//just set neuron m_activation to 0 bc we train it anyway
		neurons.push_back(new Neuron(math.rng(),Neuron::NEURON,sizes[0].second));
	}
	//biases are always at the last index
	neurons.push_back(new Neuron(1, Neuron::BIAS,Neuron::SIGMOID));
	neuronLayers.push_back(neurons);
	neurons.clear();

	//we need no biases here, bc no next layer after outputs
	//define our output neuron(s)
	for (int j = 0; j < sizes[numLayers-1].first; j++) {
		//just set neuron m_activation to 0 bc we train it anyway
		neurons.push_back(new Neuron(math.rng(), Neuron::NEURON,sizes[numLayers-1].second));
	}
	neuronLayers.push_back(neurons);
	neurons.clear();

	std::vector<std::vector<Neuron*>> hiddenLayers;
	//hidden layers (for loop is here unnecessary bc of 1 hidden layer)
	for (int i = 1; i < numLayers -1; i++) {
		hiddenLayers.resize(i);
		//iterate through neurons
		for (int j = 0; j < sizes[i].first; j++) {
			//just set neuron m_activation to 0 bc we train it anyway
			hiddenLayers[i - 1].push_back(new Neuron(math.rng(),Neuron::NEURON,sizes[i].second));
		}
		//add our bias
		hiddenLayers[i - 1].push_back(new Neuron(1, Neuron::BIAS,Neuron::SIGMOID));
	}

	for (std::vector<Neuron*> hiddenLayer : hiddenLayers){
        neuronLayers.insert(neuronLayers.end() - 1, hiddenLayer);
	}

	hiddenLayers.clear();
}

/**
 * Trains a constructed network
 *
 * @param std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData
 * @param float learning Rate as float between 0-1
 * @param float momentum as float between 0-1
 * @param int how many epochs to train on this network
 * @return 0 on sucess
 */
int Network::train(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData,float learningRate,float momentum, int epochsToTrain) {

	this->learningRate = learningRate;
	this->momentum = momentum;
    setEpochsToTrain(epochsToTrain);

	//do this epoch times
	//first we need to calculate the error
	std::vector<std::vector<Neuron*>> allOutputs;
	allOutputs.reserve(trainingData.size());

	for (int currentEpoch = 0; currentEpoch < epochsToTrain; currentEpoch++) {
        setCurrentEpoch(currentEpoch);
        if(signalFlag == 2){
            std::cout << "Interrupted trainging in Epoch: " << currentEpoch << std::endl;
            signalFlag = 0;
            return 0;
        }

		//benchmarking
		auto startEpoch = std::chrono::high_resolution_clock::now();

		std::cout << "Epoch: " << currentEpoch << std::endl;
        
		for (std::pair<std::vector<float>, std::vector<float>> &pair : trainingData) {

			//benchmarking
			auto startForward = std::chrono::high_resolution_clock::now();

			//TODO improve performance of forward propagation
			//also set activations
			//add all highest outputs from training data together
			std::vector<Neuron*> networkOuput = feedForward(pair.first);
            
			//benchmarking
			auto finishForward = std::chrono::high_resolution_clock::now();
			auto microsecondsForward = std::chrono::duration_cast<std::chrono::microseconds>(finishForward - startForward);
			averageTimeForward += microsecondsForward.count();
			timesForward++;

			//push back to later compare
			allOutputs.push_back(networkOuput);

			//benchmarking
			auto startBackprop = std::chrono::high_resolution_clock::now();

			//secondly we need to calculate m_delta for each neuron in the network (except input neurons), we are starting from behind
            //WARNING: changed to calculate input neurons as well as of implementation into CNN
			for (int i = neuronLayers.size() - 1; i > 0; i--) {
				//-1 bc we dont want to calculate for the bias
				for (int j = 0; j <= neuronLayers[i].size() - 1; j++) {
					//the networkOuput layer uses a different function
					if (i == neuronLayers.size() - 1) {
						//m_delta = f'(m_sum) * (o - t)
						neuronLayers[i][j]->setDelta(neuronLayers[i][j]->actPrime(neuronLayers[i][j]->getSum()) * (neuronLayers[i][j]->getActivation() - pair.second[j]));
					}
					else {
						float sum = 0;
						//neuronLayers[i+1][k].m_delta <- m_delta from layer before -1 bc we dont want to calculate for the bias
						for (int k = 0; k < neuronLayers[i+1].size(); k++) {
							if(neuronLayers[i + 1][k]->getType() == Neuron::NEURON)
								sum += neuronLayers[i+1][k]->getDelta() * weights[i][k][j];
						}
						//m_delta = f'(m_sum)*m_sum(prevDeltas*weights)
						neuronLayers[i][j]->setDelta(neuronLayers[i][j]->actPrime(neuronLayers[i][j]->getSum()) * sum);
					}
				}
			}

			//benchmarking
			auto finishBackprop = std::chrono::high_resolution_clock::now();
			auto microsecondsBackprop = std::chrono::duration_cast<std::chrono::microseconds>(finishBackprop - startBackprop);
			averageTimeBackprop += microsecondsBackprop.count();
			timesBackprop++;

			//std::vector<std::future<void>> threads;
			//threads.reserve(neuronLayers.size());
			//benchmarking
			auto startChange = std::chrono::high_resolution_clock::now();

			//this is ineffective
			//auto calculateChange = std::async([=] {

			//we now can calculate the weights
			for (int i = 1; i < neuronLayers.size(); i++) {
				for (int j = 0; j < neuronLayers[i].size() - 1; j++) {
					for (int k = 0; k < neuronLayers[i - 1].size(); k++) {
						//without momentum
						//float change = backProp(neuronLayers[i][j].m_delta, neuronLayers[i - 1][k].m_activation);

						//with momentum
						float change = -1 * backPropMomentum(neuronLayers[i][j]->getDelta(), neuronLayers[i - 1][k]->getActivation(), oldchange[i - 1][j][k]);
						oldchange[i - 1][j][k] = change;

						weights[i - 1][j][k] += change;
					}
				}
			}

			//});
			//threads.push_back(std::move(calculateChange));

			//for (std::future<void> &thread : threads) thread.get();

			//benchmarking
			auto finishChange = std::chrono::high_resolution_clock::now();
			auto microsecondsChange = std::chrono::duration_cast<std::chrono::microseconds>(finishChange - startChange);
			averageTimeChange += microsecondsChange.count();
			timesChange++;

            calcRMSE(pair, networkOuput);
		}
		allOutputs.clear();

		//benchmarking
		auto finishEpoch = std::chrono::high_resolution_clock::now();
		auto microsecondsEpoch = std::chrono::duration_cast<std::chrono::microseconds>(finishEpoch - startEpoch);

        std::cout << "Error: " << m_error << std::endl;
		std::cout << averageTimeForward / timesForward << " microseconds forward calculation\n";
		std::cout << averageTimeBackprop/timesBackprop << " microseconds backprop calculation\n";
		std::cout << averageTimeChange/timesChange << " microseconds change calculation\n";
		std::cout << microsecondsEpoch.count() << " microseconds currentEpoch calculation\n";
		averageTimeForward = 0;
		timesForward = 0;
		averageTimeBackprop = 0;
		timesBackprop = 0;
		averageTimeChange = 0;
		timesChange = 0;

		averageTimeEpoch += microsecondsEpoch.count();

	}

	std::cout << "Average time for one epoch: " << averageTimeEpoch / 100 << std::endl;
	std::cout << "One training data step: " << averageTimeEpoch / 100 / 10000 << std::endl;

//         zero old changes
// 	for (int i = 0; i < oldchange.size(); i++) {
// 		for (int j = 0; j < oldchange[i].size(); j++) {
// 			for (int k = 0; k < oldchange[i][j].size(); k++) {
// 				oldchange[i][j][k] = 0;
// 			}
// 		}
// 	}
	//allOutputs.clear();

	return 0;
}

std::vector<Neuron*> Network::feedForward(std::vector<float> &testData) {

	//apply testData to Input Layer
	for (size_t i = 0; i < neuronLayers[0].size(); i++) {
            if(neuronLayers[0][i]->getType() == Neuron::NEURON){
                neuronLayers[0][i]->setActivation(testData[i]);
            }
	}

	for (size_t i = 1; i < neuronLayers.size(); i++) {
            for (size_t j = 0; j < neuronLayers[i].size(); j++) {
		        if (neuronLayers[i][j]->getType() == Neuron::NEURON) {
                    neuronLayers[i][j]->calculateActivation(neuronLayers[i - 1], weights[i - 1][j]);
                }
            }
	}
	//std::cout << neuronLayers[numLayers - 1][0].m_activation << std::endl;
	//return output
	return neuronLayers[numLayers-1];
}

std::vector<float> Network::predict(std::vector<float> &testData) {

	std::vector<Neuron*> output = feedForward(testData);
	std::vector<float> values;
	values.reserve(output.size());

	for (Neuron* neuron : output) {
		values.push_back(neuron->getActivation());
	}
        output.clear();
	return values;
}

/*
*	Training algorithms
*/
float Network::backProp(float &deltaCurrent,float &activationBefore) {  
	//j -> i
	//learningrate * deltai * activationj
	float weightChange =  -1* learningRate * deltaCurrent * activationBefore;
	return weightChange;
}
//backpropagation with momentum
float Network::backPropMomentum(float deltaCurrent, float activationBefore, float &oldChange) {
    //j -> i
    //learningrate * deltai * activationj
    float weightChange = (1-momentum) * learningRate * deltaCurrent * activationBefore + momentum * oldChange;
    return weightChange;
}

/*
*	Error function(s) Note: Error is also found as Cost! These are the same things
*/
//mean quared error
void Network::calcMSE(std::vector<std::pair<std::vector<float>, std::vector<float>>> &trainingData, std::vector<std::vector<Neuron*>> &outputNeurons) {

	float error = 0;
	//ideals holds all ideal values in the training set/batch
	std::vector<std::vector<float>> ideals;

	//get our ideals list
	for (std::pair<std::vector<float>, std::vector<float>> batch : trainingData) {
		ideals.push_back(batch.second);
	}
	//mean sqared error = ((ideal1-output1)^2+(...)+(idealn-outputn)^2)/n
	//both vectors are for example 10 big so we can calculate an error. Ideals must be filles up with ideals for all neurons
	//for example ideals: {1,0,0,0,0,0,0,0,0,0} output: {0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}
	for (size_t i = 0; i < ideals.size(); i++) {
		for (size_t j = 0; j < ideals[i].size(); j++) {
			error += pow((ideals[i][j] - outputNeurons[i][j]->getActivation()), 2);
		}
	}
	//this is the mean squared error
	error = error / ideals.size();
	std::cout << "Error: " << error << std::endl;

    m_error = error;
}

//root mean sqared error
void Network::calcRMSE(std::pair<std::vector<float>, std::vector<float>> &trainingData, std::vector<Neuron*>  &outputNeurons) {

	float error = 0;

	//mean sqared error = ((ideal1-output1)^2+(...)+(idealn-outputn)^2)/n
	//both vectors are for example 10 big so we can calculate an error. Ideals must be filles up with ideals for all neurons
	//for example ideals: {1,0,0,0,0,0,0,0,0,0} output: {0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}
	for (size_t i = 0; i < trainingData.second.size(); i++) {
        error += pow((trainingData.second[i] - outputNeurons[i]->getActivation()), 2);
	}
	error = error / trainingData.second.size();
	//this is the root mean squared error
	error = sqrt(error);

	m_error = error;
}
//arctan error TODO


/**
 * Get highest value index from vector
 * @param outputValues
 * @return index of highest value
 */
int Network::highestPred(std::vector<float> &outputValues) {

	//get highest value
	std::vector<float>::iterator it = std::max_element(outputValues.begin(), outputValues.end());
	int index = std::distance(outputValues.begin(), it);

	return index;
}
/**
 * Save network weights and layout to file
 * @param filename NAme of the file to save to
 * @return 0 if successful
 */
int Network::save(std::string filename) {

	std::ofstream file(filename, std::ios::binary);

	//return if file not open
	if (!file.is_open()) {
		std::cout << "File could not be created!" << std::endl;
		return 1;
	}

        //write layer size
	file << (int) sizes.size() << "::\n";
	for (int i = 0; i < sizes.size(); i++) {
		file << (int) sizes[i].first << ":";
	}
	//write layer m_activationFunction
        file << ":\n";
        for (int i = 0; i < sizes.size(); i++) {
		file << (int) sizes[i].second << ":";
	}
        file << ":\n";
	//write weights
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				file << weights[i][j][k] << ":";
			}
		}
	}
        file << ":";
        std::cout << "Check: " << weights[0][0][0] << std::endl;
	return 0;
}
/**
 * Load network weights and layout from file
 * @param filename Name of the file to load from
 * @return 0 if successful
 */
int Network::load(std::string filename) {

    std::string delimiter = ":";

    std::ifstream file(filename, std::ios::binary);

    //return if file not open
    if (!file.is_open()) {
        std::cout << "File could not be opened!" << std::endl;
        return 1;
    }

	//read size
	std::string layers = "";
        file >> layers;
        int sizeofLayers = layers.size();
        this->numLayers = std::stoi(layers.substr(0,layers.size()-2));

        std::string layerSizes = "";
        file.seekg(sizeofLayers);
        file >> layerSizes;
        int sizeofLayerSizes = layerSizes.size();

        std::string layerModes = "";
        file.seekg(sizeofLayers+sizeofLayerSizes);
        file >> layerModes;
        int sizeofLayersModes = layerModes.size();

        std::vector<std::pair<int,int>> readSizes;

        size_t pos = 0;
        std::string token;
        while ((pos = layerSizes.find(delimiter)) != std::string::npos) {
            token = layerSizes.substr(0, pos);
            if(token == "") break;
            readSizes.push_back({std::stoi(token),0});
            layerSizes.erase(0, pos + delimiter.length());
        }

        pos = 0;
        token = "";
        int index = 0;
        while ((pos = layerModes.find(delimiter)) != std::string::npos) {
            token = layerModes.substr(0, pos);
            if(token == "") break;
            readSizes[index].second = std::stoi(token);
            layerModes.erase(0, pos + delimiter.length());
            index++;
        }
        this->sizes = readSizes;

        //read weights
        std::vector<std::vector<std::vector<float>>> readWeights;
        //redundant performance improvement
        readWeights.reserve(sizes.size());
        int y = 1;
	for (int x = 0; x < numLayers;x++) {
		if (y >= numLayers) break;
		//inititalize weight vector with random numbers
		readWeights.push_back(math.defmatrix(sizes[x].first+1, sizes[y].first));
		y++;
	}

        std::string allWeightsString = "";
        file.seekg(sizeofLayers+sizeofLayerSizes+sizeofLayersModes+10);
        file >> allWeightsString;

        std::vector<float> allWeights;

        pos = 0;
        token = "";
        while ((pos = allWeightsString.find(delimiter)) != std::string::npos) {
            token = allWeightsString.substr(0, pos);
            if(token == "") break;
            allWeights.push_back(std::stof(token));
            allWeightsString.erase(0, pos + delimiter.length());
        }

        int l = 0;
	for (int i = 0; i < readWeights.size(); i++) {
		for (int j = 0; j < readWeights[i].size(); j++) {
			for (int k = 0; k < readWeights[i][j].size(); k++) {
                            readWeights[i][j][k] = allWeights[l];
                            l++;
			}
		}
	}

        this->weights = readWeights;
        std::cout << "Check: " << weights[0][0][0] << std::endl;
    return 0;
}

/**
 * As for use as API in cnn networks
 * Only backpropagates once
 * @param t_trainingData pair of training data. The first @code{std::vector<float>} contains network inputs while the second contains ideal outputs
 * @param t_learningRate learning rate to apply to network
 * @param t_momentum momentum to apply to network
 * @return vector containing deltas of neurons
 */
std::vector<float> Network::train_once(std::pair<std::vector<float>, std::vector<float>> &t_trainingData, float t_learningRate, float t_momentum){
    learningRate = t_learningRate;
    momentum = t_momentum;
    //returned deltas of input layer
    std::vector<float> deltas;

    //TODO improve performance of forward propagation
    //also set activations
    std::vector<Neuron*> networkOutputs = feedForward(t_trainingData.first);
    /*for (size_t i = 0; i < neuronLayers[0].size(); i++) {
        if(neuronLayers[0][i]->getType() == Neuron::NEURON)
            neuronLayers[0][i]->setActivation(t_trainingData.first[i]);
    }*/

    //secondly we need to calculate m_delta for each neuron in the network (except input neurons), we are starting from behind
    //WARNING: changed to calculate input neurons as well after implementation into CNN
    for (int i = neuronLayers.size() - 1; i >= 0; i--) {
        //neuronLayers[i].size()-1 because we dont want to calculate for the bias
        for (size_t j = 0; j < neuronLayers[i].size(); j++) {
            //the output layer uses a different function
            if (i == neuronLayers.size() - 1) {
                //m_delta = f'(m_sum) * (o - t)
                neuronLayers[i][j]->setDelta(neuronLayers[i][j]->actPrime(neuronLayers[i][j]->getSum()) * (neuronLayers[i][j]->getActivation() - t_trainingData.second[j]));
            }
            else {
                float sum = 0;
                //neuronLayers[i+1][k].m_delta <- m_delta from layer before -1 bc we dont want to calculate for the bias
                for (size_t k = 0; k < neuronLayers[i+1].size(); k++) {
                    if(neuronLayers[i + 1][k]->getType() == Neuron::NEURON)
                        sum += neuronLayers[i+1][k]->getDelta() * weights[i][k][j];
                }
                //m_delta = f'(m_sum)*m_sum(prevDeltas*weights)
                neuronLayers[i][j]->setDelta(neuronLayers[i][j]->actPrime(neuronLayers[i][j]->getSum()) * sum);
                if(i == 0){
                    deltas.push_back(neuronLayers[i][j]->getDelta());
                }
            }
        }
    }

    //std::vector<std::future<void>> threads;
    //threads.reserve(neuronLayers.size());

    //this is ineffective
    //auto calculateChange = std::async([=] {

    //we now can calculate the weights
    for (size_t i = 1; i < neuronLayers.size(); i++) {
        for (size_t j = 0; j < neuronLayers[i].size() - 1; j++) {
            for (size_t k = 0; k < neuronLayers[i - 1].size(); k++) {
                //without momentum
                //float change = backProp(neuronLayers[i][j].m_delta, neuronLayers[i - 1][k].m_activation);

                //with momentum
                float change = -1 * backPropMomentum(neuronLayers[i][j]->getDelta(), neuronLayers[i - 1][k]->getActivation(), oldchange[i - 1][j][k]);
                //apply weight change
                oldchange[i - 1][j][k] = change;
                weights[i - 1][j][k] += change;
            }
        }
    }

    calcRMSE(t_trainingData,networkOutputs);

    //});
    //threads.push_back(std::move(calculateChange));

    //for (std::future<void> &thread : threads) thread.get();

    return deltas;
 }

float Network::getLearningRate() const {
    return learningRate;
}

void Network::setLearningRate(float learningRate) {
    Network::learningRate = learningRate;
}

float Network::getMomentum() const {
    return momentum;
}

void Network::setMomentum(float momentum) {
    Network::momentum = momentum;
}

int Network::getEpochsToTrain() const {
    return epochsToTrain;
}

void Network::setEpochsToTrain(int epochsToTrain) {
    Network::epochsToTrain = epochsToTrain;
}

int Network::getCurrentEpoch() const {
    return currentEpoch;
}

void Network::setCurrentEpoch(int currentEpoch) {
    Network::currentEpoch = currentEpoch;
}
