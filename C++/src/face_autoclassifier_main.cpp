#include "nn/Network.hpp"
#include "deb.hpp"
#include <iomanip>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using nn::flt_t;
using nn::Sample;

constexpr size_t IMAGE_SIZE = 64;

constexpr auto compare = [](const std::vector<flt_t>& expectedOutputs, const std::vector<flt_t>& actualOutputs){
    return false;
};

std::pair<std::vector<Sample>, std::vector<Sample>> getImages(size_t trainCount = 12233, size_t testCount = 1000) {
    std::vector<std::string> imageFilenames;
    std::string line;
    std::ifstream filenamesFile{"./lfw-deepfunneled/filenames.txt"};
    while(getline(filenamesFile, line)) {
        imageFilenames.push_back(line);
    }

    if (trainCount + testCount > imageFilenames.size()) {
        throw std::runtime_error{"Required image count (" + std::to_string(trainCount + testCount)
            + ") is smaller than dataset size (" + std::to_string(imageFilenames.size()) + ")"};
    }
    std::random_shuffle(imageFilenames.begin(), imageFilenames.end());
    imageFilenames.resize(trainCount + testCount);

    std::vector<Sample> train, test;
    for (size_t i = 0; i != trainCount + testCount; ++i) {
        std::string file = "./lfw-deepfunneled/" + imageFilenames[i];
        deb("Reading file", i, (i < trainCount ? "(train)" : "(test) "), file);

        int width, height, channels;
        uint8_t* rgb_image = stbi_load(file.c_str(), &width, &height, &channels, 3);

        if (width != IMAGE_SIZE || height != IMAGE_SIZE || channels != 3) {
            throw std::runtime_error{"Read image has invalid size (not " + std::to_string(IMAGE_SIZE) + "x" + std::to_string(IMAGE_SIZE)
                + "x3) for file " + file + ": " + std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(channels)};
        }

        std::vector<nn::flt_t> image;
        image.reserve(IMAGE_SIZE*IMAGE_SIZE*3);
        for (size_t j = 0; j != IMAGE_SIZE*IMAGE_SIZE*3; ++j) {
            image.push_back(rgb_image[j] / (flt_t) 255.0);
        }

        (i < trainCount ? train : test).push_back(std::move(image));
        stbi_image_free(rgb_image);
    }

    return {train, test};
}

int main() {
	nn::Network net{nn::fastSigmoid, nn::crossEntropyCost};
    std::cout<<"Loading network..."<<std::flush;
	std::ifstream fin{"network_images.txt"};
    fin >> net;
    fin.close();
    std::cout<<"\rLoaded            \n";
    auto [trainImages, testImages] = getImages(11000, 2233);

	net.momentumSGD(trainImages, 1, 50, 0.1 , 6.0, 0.5 , testImages, std::cout, compare);

	std::ofstream fout{"network_images_1.txt"};
	fout << net;
	fout.close();
}