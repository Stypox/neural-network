/*
 * Node.hpp
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

struct Node {
	struct Param {
		flt_t value;
		flt_t derivative = 0;

		Param(flt_t v) : value{v} {}
		Param& operator=(flt_t v) { value = v; return *this; }

		inline void resetDerivative() { derivative = 0; }
		inline operator flt_t() { return value; }
	};

	flt_t m_value;

	// the following members are not used in input nodes
	Param m_bias;
	std::vector<Param> m_weights; // weights coming from inputs

	// the following members are needed to calculate derivatives
	// TODO remove assignments to m_valueBeforeSig when just calculating (not training)
	// TODO replace with s'(m_valueBeforeSig)
	flt_t m_valueBeforeSig;
	flt_t m_derivativeFromHereOn;


	Node(size_t inputCount);

	void setValueDirectly(flt_t value);
	void setValueFromSum(flt_t sumInputs);

	Param& weightFrom(size_t nodeA);
};

} /* namespace nn */

#endif /* NODE_HPP_ */
