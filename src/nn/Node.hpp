/*
 * Node.h
 *
 *  Created on: May 18, 2019
 *      Author: stypox
 */

#ifndef NODE_HPP_
#define NODE_HPP_

#include <vector>
#include <ostream>
#include "shared.hpp"

namespace nn {

class Node {
	friend class Network;

	flt_t m_value;
	std::vector<flt_t> m_weights, m_outputs;

	// things for the network
	flt_t m_sumInputs;
	std::vector<flt_t> m_weightDerivatives;
	flt_t m_derivativeSoFar;
public:
	Node(size_t outputCount);
	Node(std::vector<flt_t> weights);

	void setValueDirectly(flt_t value);
	void setValueFromSum(flt_t sumInputs);

	flt_t weightTo(size_t nodeB) const;
};

} /* namespace nn */

#endif /* NODE_HPP_ */
