#ifndef _NN_NODE_HPP_
#define _NN_NODE_HPP_

#include <vector>
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

	Node(size_t inputCount);
};

} /* namespace nn */

#endif /* _NN_NODE_HPP_ */
