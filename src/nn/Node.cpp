/*
 * Node.cpp
 *
 *  Created on: May 18, 2019
 *      Author: stypox
 */

#include "Node.hpp"
#include <algorithm>

namespace nn {

Node::Node(size_t outputCount) :
		m_value{}, m_weights(outputCount),
		m_outputs(outputCount),
		m_sumInputs{}, m_weightDerivatives(outputCount),
		m_derivativeSoFar{} {
	std::generate(m_weights.begin(), m_weights.end(), nn::random);
}

Node::Node(std::vector<flt_t> weights) :
		m_value{}, m_weights{weights},
		m_outputs(weights.size()),
		m_sumInputs{}, m_weightDerivatives(weights.size()),
		m_derivativeSoFar{} {}

void Node::setValueDirectly(flt_t value) {
	m_value = value;
	for (size_t i = 0; i < m_weights.size(); ++i) {
		m_outputs[i] = m_weights[i]*m_value;
	}
}
void Node::setValueFromSum(flt_t sumInputs) {
	m_sumInputs = sumInputs;
	setValueDirectly(sig(sumInputs));
}

flt_t Node::weightTo(size_t nodeB) const {
	return m_weights[nodeB];
}

} /* namespace nn */
