#include "Node.hpp"
#include <algorithm>

namespace nn {

Node::Node(size_t inputCount) :
		bias{nn::random()}, weights{},
		z{}, a{},
		error{}, weightsNabla(inputCount),
		accBiasNabla{}, accWeightsNabla(inputCount) {
	std::generate_n(std::back_inserter(weights), inputCount, nn::random);
}

} /* namespace nn */
