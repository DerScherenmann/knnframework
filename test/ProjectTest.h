//
// Created by Robert Nickel on 1/8/23.
//

#ifndef KNNFRAMEWORK_PROJECTTEST_H
#define KNNFRAMEWORK_PROJECTTEST_H

    typedef unsigned char uchar;

    bool sortNeuron(Neuron i, Neuron j) { return (i.getActivation() < j.getActivation()); }

    int readLabelsAndImages(int number_of_images, int number_of_labels);
    unsigned char* readBMP(std::string filename);

    int main(int argc, char **argv);

#endif //KNNFRAMEWORK_PROJECTTEST_H
