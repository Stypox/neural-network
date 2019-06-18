#include "Node.hpp"
#include <algorithm>

namespace nn {

Node::Node(size_t inputCount) :
		bias{}, weights(inputCount),
		z{}, a{},
		error{}, weightsNabla(inputCount),
		accBiasNabla{}, accWeightsNabla(inputCount),
		biasVelocity{}, weightsVelocity(inputCount) {}

std::istream& operator>>(std::istream& in, Node& node) {
	in >> node.bias;

	size_t weightsSize;
	in >> weightsSize;
	node.weights.resize(weightsSize);
	node.weightsNabla.resize(weightsSize);
	node.accWeightsNabla.resize(weightsSize);
	node.weightsVelocity.resize(weightsSize);

	for(size_t yFrom = 0; yFrom != weightsSize; ++yFrom) {
		in >> node.weights[yFrom];
	}

	return in;
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
	out << node.bias << " " << node.weights.size() << " ";

	for(size_t yFrom = 0; yFrom != node.weights.size(); ++yFrom) {
		out << node.weights[yFrom] << " ";
	}

	return out;
}

} /* namespace nn */
