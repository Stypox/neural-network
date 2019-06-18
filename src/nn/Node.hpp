#ifndef _NN_NODE_HPP_
#define _NN_NODE_HPP_

#include <vector>
#include <istream>
#include <ostream>
#include "utils.hpp"

namespace nn {

struct Node {
	flt_t bias;
	std::vector<flt_t> weights;

	flt_t z, a; // a = sigmoid(z)

	flt_t error; // == biasNabla
	std::vector<flt_t> weightsNabla;

	// accumulated nablas
	flt_t accBiasNabla;
	std::vector<flt_t> accWeightsNabla; // remove this?

	// velocities
	flt_t biasVelocity;
	std::vector<flt_t> weightsVelocity;

	Node(size_t inputCount);

	friend std::istream& operator>>(std::istream& in, Node& node);
	friend std::ostream& operator<<(std::ostream& out, const Node& node);
};

} /* namespace nn */

#endif /* _NN_NODE_HPP_ */
