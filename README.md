# Neural Network library written in cpp

## Usage:

### First off design your network structure:

For example: 784 input neurons, 30 hidden neurons and 10 output neurons, all using the sigmoid function

`std::vector<std::pair<int,int>> layers = {{784,Neuron::modes::SIGMOID},{30,Neuron::modes::SIGMOID}, {10,Neuron::modes::SIGMOID}};`

### Then create a network:

`Network net = Network(layers);`
