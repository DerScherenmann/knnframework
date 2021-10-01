# Neural Network library written in cpp

## Usage:

### First off design your network structure:

For example: 784 input neurons, 30 hidden neurons and 10 output neurons, all using the sigmoid function

`std::vector<std::pair<int,int>> layers = {{784,Neuron::modes::SIGMOID},{30,Neuron::modes::SIGMOID}, {10,Neuron::modes::SIGMOID}};`

### Then create a network:

`Network network = Network(layers);`

### Provide the network with adequate training data and parameters and start training:

`network.train(trainingData,0.05,0.4,100);`

### If training has successfully completed, read the output values, after inputting some data:

`std::vector<float> predictions = network.predict(testData);`

Get the highest prediction, with wich you can read from `predictions`:

`int index = network.highestPred(predictions);`


## Tips and Tricks:

You may experience some bugs or other weird stuff. Please open an issue or open a pull request if needed.
