#include "nn/Network.hpp"
#include "deb.hpp"
#include <iomanip>
#include <vector>
#include <fstream>
#include <random>

using nn::flt_t;
using nn::Sample;

constexpr auto compare = [](const std::vector<flt_t>& expectedOutputs, const std::vector<flt_t>& actualOutputs){
	size_t ei = std::distance(expectedOutputs.begin(), std::max_element(expectedOutputs.begin(), expectedOutputs.end()));
	size_t ai = std::distance(actualOutputs.begin(), std::max_element(actualOutputs.begin(), actualOutputs.end()));
	return ei == ai;
};

std::vector<Sample> readImages(std::string imagesFilename, std::string labelsFilename = "") {
	auto getUint32 = [](std::ifstream& file) {
		uint32_t number = 0;
		for(int b = 0; b != 4; ++b) {
			number <<= 8;
			number |= file.get();
		}
		return number;
	};

	// read images file, that contains the pixels
	std::ifstream imagesFile{imagesFilename, std::ios::binary};
	imagesFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	getUint32(imagesFile); // ignore magic number
	uint32_t imageCount = getUint32(imagesFile);
	uint32_t rows = getUint32(imagesFile);
	uint32_t columns = getUint32(imagesFile);

	std::vector<Sample> images;
	for(uint32_t i = 0; i != imageCount; ++i) {
		images.push_back(Sample{});
		for(uint32_t j = 0; j != rows * columns; ++j) {
			images.back().inputs.push_back((flt_t)imagesFile.get() / (flt_t)256.0);
		}
	}

	// read labels file, that contains the numbers
	if(labelsFilename != "") {
		std::ifstream labelsFile{labelsFilename, std::ios::binary};
		labelsFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

		getUint32(labelsFile); // ignore magic number
		if(uint32_t otherImageCount = getUint32(labelsFile); otherImageCount != imageCount)
			throw std::runtime_error{"Files have different image counts: " + std::to_string(imageCount) + " and " + std::to_string(otherImageCount)};

		for(uint32_t i = 0; i != imageCount; ++i) {
			uint8_t number = labelsFile.get();
			images[i].expectedOutputs.resize(10, 0.0);
			images[i].expectedOutputs[number] = 1.0;
		}
	}

	return images;
}

int main() {
	//std::cout << std::fixed << std::setprecision(1);

	const auto trainImages = readImages("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	const auto testImages = readImages("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

	nn::Network net{{784, 100, 10}, nn::sigmoid, nn::crossEntropyCost};
	//nn::Network net{nn::crossEntropyCost};

	// std::ifstream fin{"network.txt"};
	// fin >> net;
	// fin.close();

	net.momentumSGD(trainImages,  2, 10, 0.15, 4.0, 0.8 , testImages, std::cout, compare);
	net.momentumSGD(trainImages,  3, 10, 0.1 , 4.0, 0.3 , testImages, std::cout, compare);
	net.momentumSGD(trainImages,  5, 10, 0.1 , 4.5, 0.1 , testImages, std::cout, compare);
	net.momentumSGD(trainImages, 10, 10, 0.08, 4.5, 0.02, testImages, std::cout, compare);
	net.        SGD(trainImages, 25, 15, 0.07, 5.0,       testImages, std::cout, compare);
	net.        SGD(trainImages, 25, 20, 0.06, 5.0,       testImages, std::cout, compare);

	// for(auto&& image : testImages) {
	// 	auto eo = image.expectedOutputs;
	// 	auto ao = net.calculate(image.inputs);

	// 	size_t ei = std::distance(eo.begin(), std::max_element(eo.begin(), eo.end()));
	// 	size_t ai = std::distance(ao.begin(), std::max_element(ao.begin(), ao.end()));

	// 	if (ei != ai) {
	// 		printImage(std::cout, image);
	// 		std::cout << "-------- " << ai << " -------- <<-- WRONG\n\n";
	// 	}
	// }

	std::ofstream fout{"network.txt"};
	fout << net;
	fout.close();

//	nn::Network net{5,3,3,4};
//
//	std::cout << std::fixed << std::setprecision(1);
//
//	std::vector<nn::flt_t>
//		giIn{1.,1.,0.,0.,1.}, giOut{1.,0.,0.,0.},
//		elIn{1.,1.,1.,1.,1.}, elOut{0.,1.,0.,0.},
//		goIn{0.,0.,1.,1.,1.}, goOut{0.,0.,1.,0.},
//		anIn{0.,0.,0.,0.,1.}, anOut{0.,0.,0.,1.};
//
//	deb("Before training:");
//	deb("Giraffe ", net.calculate(giIn), "-", giOut);
//	deb("Elephant", net.calculate(elIn), "-", elOut);
//	deb("Goat    ", net.calculate(goIn), "-", goOut);
//	deb("Ant     ", net.calculate(anIn), "-", anOut);
//
//	for (int i = 0; i < 1000000; ++i) {
//		net.train(giIn, giOut); // Giraffe
//		net.train(elIn, elOut); // Elephant
//		net.train(goIn, goOut); // Goat
//		net.train(anIn, anOut); // Ant
//	}
//
//	deb("\nAfter training:");
//	deb("Giraffe ", net.calculate(giIn), "-", giOut);
//	deb("Elephant", net.calculate(elIn), "-", elOut);
//	deb("Goat    ", net.calculate(goIn), "-", goOut);
//	deb("Ant     ", net.calculate(anIn), "-", anOut);
}
