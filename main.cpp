#include "main.h"
#include <iostream>
#include "network.h"
#include "neuron.h"
#include "mathhelper.h"
#include <fstream>
#include <thread>
#include <map>
#include <algorithm>
#include <sstream>

#include <thread>

#define __GPU

typedef unsigned char uchar;

bool sortNeuron(Neuron i, Neuron j) { return (i.activation < j.activation); }

std::vector<std::vector<float>> images;
uchar* labels;
std::vector<std::vector<float>> testimages;
uchar* testlabels;

int readStuff(int number_of_images, int number_of_labels);
unsigned char* readBMP(std::string filename);

int main(int argc, char* argv[]) {
    
    printf("Neural Networks HomeOffice created on 20.05.2020 22:56 GMT+1\n\n\nStarting...\n");
    reinitNetwork:
    //our network
    //image recognition
    std::vector<std::pair<int,int>> layers = {{784,Neuron::modes::SWISH},{30,Neuron::modes::SWISH}, {10,Neuron::modes::SWISH},{10,Neuron::modes::SIGMOID}};
    Network net = Network(layers);
    
    while (1) {

	printf("Operating Modes are:\n[0] Load database into memory\n[1] Train network with MNIST databse\n[2] Test Network with MNIST images\n[3] Use your own hand drawn image\n");
	printf("[4] Save network to file\n[5] Load network from file\n[6] Reinitialize neural network\n");

	char readChar = getchar();
		
	if (readChar == '0') {
            
            if (readStuff(60000,60000)) {
                std::cout << "Success!" << std::endl;
            }

		}
        if (readChar == '1') {

            /*
            *   Start training
            */
            int numImages = 100;
            
            std::vector<std::pair<std::vector<float>, std::vector<float>>> trainingData;
            //lets use 100 images   
            trainingData.resize(numImages);
            for (int i = 0; i < numImages; i++) {    
                //0-255 to 0-1
                //float value = images[i][j] / 255;
                for (int j = 0; j < images[i].size(); j++) {
                    images[i][j] = images[i][j] / 255;
                }
                trainingData[i].first = images[i];
                std::vector<float> temp;
                temp.resize(10);
                for (int j = 0; j < 10; j++) {
                    if (labels[i] == j) {
                        temp[j] = 1;
                    }
                    else {
                        temp[j] = 0;
                    }
                }
                trainingData[i].second = temp;
            }
                
            net.train(trainingData,0.05,0.4,100);
            //start training with images
            trainingData.clear();      
        }
        if (readChar == '2') {
            
            Math math;
            int testindex = math.rand_in_range(0, 10000);
            std::cout << testindex << " in " << testimages.size() << std::endl;
            std::vector<float> testData = testimages[testindex];
            std::cout << "Data: " << (int) (testlabels[testindex]) << std::endl;

            //predict
            std::vector<float> outputValues = net.predict(testData);
            //get highest value
            int index = net.highestPred(outputValues);
            std::cout << "Prediction: " << index << " confidence: " << outputValues[index] << std::endl;
            
            outputValues.clear();

        }
        if (readChar == '3') {

            /*
            *   Parse Hand drawn bitmaps
            */
            
            std::cout << "Enter name of image below: " << std::endl;
            std::string filename;
            std::cin >> filename;

            unsigned char* dataGrey = readBMP(filename);

            if (dataGrey != NULL) {
                //thicc
                for (int i = 28; i > 0; i--) {
                    for (int j = 0; j < 28; j++) {
                        if (dataGrey[(i * 28 + j)] == 0) {
                            std::cout << "1";
                        }
                        else {
                            if (255 - (int)dataGrey[(i * 28 + j)] == 0) {
                                std::cout << "0";
                            }
                            else {
                                std::cout << (255 - (int)dataGrey[(i * 28 + j)])/255;
                            }
                        }
                    }
                    std::cout << std::endl;
                }

                //invert data and parse it between 0 and 1
                std::vector<float> testData;
                testData.resize(784);
                for (int i = 0; i < 784; i++) {
                    testData[i] = (255 - dataGrey[i]) / 255;
                }
                
                Math math;
                //predict
                std::vector<float> outputValues = net.predict(testData);
                //get highest value
                int index = net.highestPred(outputValues);
                std::cout << "Prediction: " << index << " confidence: " << outputValues[index] << std::endl;
                outputValues.clear();
            }

        }
        if (readChar == '4') {
           /*
           *    Save network to file
           */
            std::string filename;

            std::cout << "File to save to: " << std::endl;
            std::cin >> filename;

            net.save(filename);

        }
        if (readChar == '5') {
            /*
            *   Load network from file            
            */
            std::string filename;

            std::cout << "File to load from: " << std::endl;
            std::cin >> filename;

            net.load(filename);
        }
        if (readChar == '6') {
            /*
            *   Reinit neuronal network
            */
            //Network net = Network(std::vector<int>{784, 30, 10},net.SIGMOID);
        }

		getchar();
	}
	
}

int readStuff(int number_of_images,int number_of_labels) {

    Math math;

    /*
    * Open training images
    */
    std::ifstream file("train-images.idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
        std::cout << "Opening images..." << std::endl;
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = math.reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = math.reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = math.reverseInt(n_cols);
        std::cout << "Amount: " << number_of_images << " Rows: " << n_rows << " Cols: " << n_cols << std::endl;
        //std::thread imageThread = std::thread([number_of_images,n_rows,n_cols,file,images] {
        std::vector<float> test;
        images.reserve(number_of_images);
        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    test.push_back(temp);
                }
            }
            images.push_back(test);
            test.clear();
        }
        //});
        file.close();
    }else{
        std::cout << "File not found!" << std::endl;
    }
    /*
    * Open training labels
    */
    file.open("train-labels.idx1-ubyte", std::ios::binary);
    if (file.is_open()) {
        
        std::cout << "Opening labels..." << std::endl;
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);

        if (magic_number != 2049) return 1;

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = math.reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        labels = _dataset;
        file.close();
    }
    /*
    * Open Test Images
    */
    file.open("t10k-images.idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
        std::cout << "Opening testing images..." << std::endl;
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = math.reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = math.reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = math.reverseInt(n_cols);
        testimages.reserve(10000);
        //std::thread imageThread = std::thread([number_of_images,n_rows,n_cols,file,images] {
        std::vector<float> test;
        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    test.push_back(temp);
                }
            }
            testimages.push_back(test);
            test.clear();
        }
        //});
        file.close();
    }
    /*
    * Open Test Labels
    */
    file.open("t10k-labels.idx1-ubyte", std::ios::binary);
    if (file.is_open()) {

        std::cout << "Opening test labels..." << std::endl;
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = math.reverseInt(magic_number);

        if (magic_number != 2049) return 1;

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = math.reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        testlabels = _dataset;
    }

    return 0;
}

unsigned char* readBMP(std::string filename)
{
    int i;
    FILE* f = fopen(filename.c_str(), "rb");
    unsigned char info[54];

    if (f == NULL) {
        std::cout << "File not Found!" << std::endl;
        return NULL;
    }

    // read the 54-byte header
    fread(info, sizeof(unsigned char), 54, f);

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    // allocate 3 bytes per pixel
    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size];

    unsigned char* dataGrey = new unsigned char[size];

    // read the rest of the data at once
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);

    for (i = 0; i < size; i += 3)
    {
        // flip the order of every 3 bytes
        unsigned char tmp = data[i];
        data[i] = data[i + 2];
        data[i + 2] = tmp;
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            dataGrey[(i * width + j)] = 0.3 * data[3 * (i * width + j) + 1] + 0.59 * data[3 * (i * width + j) + 1] + 0.11 * data[3 * (i * width + j) + 1];
        }
    }

    return dataGrey;
}

/*
FILE* file = fopen(filename.c_str(),"rb");

            //read header field
            unsigned char* field = (unsigned char*) malloc(2);
            fseek(file,0,0);
            fread(field,sizeof field,1,file);

            //read size
            unsigned char* size = (unsigned char*) malloc(4);
            fseek(file,2,0);
            fread(size, sizeof size, 1, file);

            //read starting adress
            unsigned char* startingAdress = (unsigned char*) malloc(4);
            fseek(file, 10, 0);
            fread(startingAdress, sizeof startingAdress, 1, file);

            //read DIB header
            unsigned char* headerSize = (unsigned char*)malloc(4);
            fseek(file, 14,0);
            fread(headerSize, sizeof headerSize, 1, file);

            //read width
            unsigned char* width = (unsigned char*) malloc(4);
            fseek(file, 18, 0);
            fread(width, sizeof width, 1, file);

            //read height
            unsigned char* height = (unsigned char*) malloc(4);
            fseek(file, 22, 0);
            fread(height, sizeof height, 1, file);

            int widthVal = ((int) *width);
            int heightVal = ((int)*height);

            int dataSize = widthVal*heightVal * 3;

            //read array
            unsigned char* data = (unsigned char*)malloc(dataSize);
            fseek(file, (int) *startingAdress, 0);
            fread(data, dataSize, 1, file);

            std::cout << std::hex << * field;
            std::cout << std::endl;
            std::cout << std::hex << (int) *size;
            std::cout << std::endl;
            std::cout << std::hex << (int) *startingAdress;
            std::cout << std::endl;
            std::cout << std::hex << (int) *headerSize;
            std::cout << std::endl;
            std::cout << std::hex << (int) *width;
            std::cout << std::endl;
            std::cout << std::hex << (int) *height;
            std::cout << std::endl;

            for (int i = 0; i < dataSize; i += 3)
            {
                // flip the order of every 3 bytes
                unsigned char tmp = data[i];
                data[i] = data[i + 2];
                data[i + 2] = tmp;
            }

            for (int i = 0; i < heightVal *3; i += 3) {
                for (int j = 0; j < widthVal *3; j += 3) {
                    if (((int) data[j * (i + 1)]) == 0) {
                        std::cout << "00";
                    }
                    std::cout << (int) data[j * (i + 1)];
                }
                std::cout << std::endl;
            }
*/
