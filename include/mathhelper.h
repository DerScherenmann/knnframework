#pragma once
#include "math.h"
#include <random>
#include <vector>
class Math {
    public:

        float rng();
        float rand_in_range(float min, float max);
        size_t rand_bias(size_t min,size_t max);

        /**********************************************************************/
        /*** sigmoid.c:  This code contains the function routine            ***/
        /***             sigmoid() which performs the unipolar sigmoid      ***/
        /***             function for backpropagation neural computation.   ***/
        /***             Accepts the input value x then returns it's        ***/
        /***             sigmoid value in float.                            ***/
        /***                                                                ***/
        /***  function usage:                                               ***/
        /***       float sigmoid(float x);                                  ***/
        /***           x:  Input value                                      ***/
        /***                                                                ***/
        /***  Written by:  Kiyoshi Kawaguchi                                ***/
        /***               Electrical and Computer Engineering              ***/
        /***               University of Texas at El Paso                   ***/
        /***  Last update:  09/28/99  for version 2.0 of BP-XOR program     ***/
        /**********************************************************************/

        float sigmoid(float x) {
                float exp_value;
                float return_value;

                /*** Exponential calculation ***/
                exp_value = exp((double)-x);

                /*** Final sigmoid value ***/
                return_value = 1 / (1 + exp_value);

                return return_value;
        }
        float sigmoidPrime(float x) {
                float return_value;

                return_value = sigmoid(x) * (1 - sigmoid(x));

                return return_value;
        }
        float swish(float x) {
                float return_value = 0;

                return_value = x * sigmoid(x);

                return return_value;
        }
        float swishPrime(float x) {
                float return_value;

                return_value =  sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x));

                return return_value;
        }

        //equivalent to randn from numpy/, returns a vector matrix with dimensions y*x (i think)
        std::vector<std::vector<float>> matrix(int y, int x) {
            std::vector<std::vector<float>> output;
            std::vector<float> data;
            for (int k = 0; k < x; k++) {
                for (int i = 0; i < y; i++) {
                    data.push_back(rng());
                }
                output.push_back(data);
                data.clear();
            }
            return output;
        }

        /**
         * Creates a vector filled with random distribution values
         * @param length final length of the vector
         * @return vector with random distribution
         */
        std::vector<float> defVector(int length){
            std::vector<float> rngVector;
            for(int i = 0;i < length;i++){
                rngVector.emplace_back(rng() / sqrt(length));
            }
            return rngVector;
        }

        std::vector<std::vector<float>> defmatrix(int y, int x) {
            std::vector<std::vector<float>> output;
            std::vector<float> data;
            for (int k = 0; k < x; k++) {
                for (int i = 0; i < y; i++) {
                    data.push_back(rng() / sqrt(x));
                }
                output.push_back(data);
                data.clear();
            }
            return output;
        }

        int reverseInt(int i) {
                unsigned char c1, c2, c3, c4;
                c1 = i & 255;
                c2 = (i >> 8) & 255;
                c3 = (i >> 16) & 255;
                c4 = (i >> 24) & 255;
                return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        }
};
