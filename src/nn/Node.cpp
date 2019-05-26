/*
 * Node.cpp
 *
 *  Created on: May 18, 2019
 *      Author: stypox
 */

#include "Node.hpp"
#include <algorithm>

namespace nn {

Node::Node(size_t inputCount) :
		m_value{}, m_bias{nn::random()},
		m_weights{}, m_valueBeforeSig{},
		m_derivativeFromHereOn{} {
	std::generate_n(std::back_inserter(m_weights), inputCount, nn::random);
}

void Node::setValueDirectly(flt_t value) {
	m_value = value;
}
void Node::setValueFromSum(flt_t sumInputs) {
	m_valueBeforeSig = sumInputs + m_bias;
	m_value = sig(m_valueBeforeSig);
}

Node::Param& Node::weightFrom(size_t nodeA) {
	return m_weights[nodeA];
}

} /* namespace nn */
