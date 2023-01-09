/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "../include/mathhelper.h"
#include <random>
#include <vector>
//Gaussian distribution
std::mt19937 generator(std::random_device{}());
float Math::rng() {
    return tanh(std::normal_distribution<float>{}(generator));
}

float Math::rand_in_range(float min, float max) {
    return std::uniform_real_distribution<float>{min, max}(generator);
}

size_t Math::rand_bias(size_t min,size_t max){
    return std::uniform_int_distribution<size_t>{min, max}(generator);
}