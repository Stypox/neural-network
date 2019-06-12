/*
 * Network.cpp
 *
 *  Created on: May 17, 2019
 *      Author: stypox
 */

#include "Network.hpp"

#include <numeric>
#include <cmath>

namespace nn {

template<bool training>
void Network::updateLayer(const size_t x) {
	for(auto&& node : m_nodes[x]) {
		flt_t sum = 0;
		for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
			sum +=
				node.weightFrom(yFrom)
				* m_nodes[x-1][yFrom].m_value;
		}

		if constexpr(training) {
			node.setValueFromSumPrepareTraining(sum);
		}
		else {
			node.setValueFromSum(sum);
		}
	}
}
template<bool training>
void Network::updateOutputs(const std::vector<flt_t>& inputs) {
	for(size_t y = 0; y != m_nodes.front().size(); ++y)
		m_nodes[0][y].setValueDirectly(inputs[y]);
	for(size_t x = 1; x != m_nodes.size(); ++x)
		updateLayer<training>(x);
}

flt_t Network::performance(const std::vector<flt_t>& expectedOutputs) {
	flt_t squaresSum = 0;
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		squaresSum += std::pow(m_nodes.back()[y].m_value-expectedOutputs[y], 2);
	}

	return squaresSum;
}

void Network::resetCurrentCostDerivatives() {
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			node.m_bias.resetCurrentCostDerivative();
			for(auto&& weight : node.m_weights) {
				weight.resetCurrentCostDerivative();
			}
		}
	}
}
void Network::resetTotalCostDerivatives() {
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			node.m_bias.resetTotalCostDerivative();
			for(auto&& weight : node.m_weights) {
				weight.resetTotalCostDerivative();
			}
		}
	}
}
void Network::resetError() {
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			node.m_error = 0;
		}
	}
}

void Network::genError(const size_t consideredOutput, const flt_t outputDelta) {
	resetError(); // this allows to use +=

	m_nodes.back()[consideredOutput].m_error =
			m_nodes.back()[consideredOutput].m_sigDerivValue
			* outputDelta;

	// ignore input layer; second layer only gets the m_error
	for(size_t x = m_nodes.size()-1; x != 1; --x) {
		for(auto&& node : m_nodes[x]) {
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x-1][yFrom].m_error +=
						m_nodes[x-1][yFrom].m_sigDerivValue
						* node.m_error
						* node.weightFrom(yFrom);
			}
		}
	}
}
void Network::addWeightDerivatives() {
	/*
		dP                                                     dO
		--  =  accumulateForEveryO( actualMinusExpected(O)  *  -- )
		dw                                                     dw

		dO
		--  =  nodeBeforeWeightValue(w)  *  error(O)
		dw

		P is the performance, w the weight, O every output node
		every actualMinusExpected(O)*dO/dw can be calculated separately and summed
	*/

	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				node.weightFrom(yFrom).currentCostDerivative +=
						node.m_error
						* m_nodes[x-1][yFrom].m_value;
			}
		}
	}
}
void Network::addBiasDerivatives() {
	/*
		dP                                                     dO
		--  =  accumulateForEveryO( actualMinusExpected(O)  *  -- )
		db                                                     db

		dO
		--  =  error(O)
		db

		P is the performance, b the bias, O every output node
		every actualMinusExpected(O)*dO/db can be calculated separately and summed
	*/

	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			node.m_bias.currentCostDerivative += node.m_error;
		}
	}
}

void Network::addDerivativesToCostDerivatives() {
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			node.m_bias.addDerivativeToCostDerivative();
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				node.weightFrom(yFrom).addDerivativeToCostDerivative();
			}
		}
	}
}

void Network::scaleAndApplyCostDerivatives(const size_t numberOfSamples, const flt_t eta) {
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x]) {
			node.m_bias.scaleAndApplyCostDerivative(numberOfSamples, eta);
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				node.weightFrom(yFrom).scaleAndApplyCostDerivative(numberOfSamples, eta);
			}
		}
	}
}

Network::Network(const std::initializer_list<size_t>& dimensions) {
	m_nodes.push_back({});
	for(size_t y = 0; y != *(dimensions.end()-1); ++y) {
		// inputs have no input-connections
		m_nodes.back().push_back(Node{0});
	}

	for(size_t x = 1; x != dimensions.size(); ++x) {
		m_nodes.push_back({});
		for(size_t y = 0; y != dimensions.begin()[x]; ++y) {
			m_nodes.back().push_back(Node{dimensions.begin()[x-1]});
		}
	}
}

std::vector<flt_t> Network::calculate(const std::vector<flt_t>& inputs) {
	updateOutputs<false>(inputs);
	std::vector<flt_t> result;
	for(size_t i = 0; i != m_nodes.back().size(); ++i)
		result.push_back(m_nodes.back()[i].m_value);
	return result;
}

flt_t Network::cost(const std::vector<Sample>::const_iterator& samplesBegin, const std::vector<Sample>::const_iterator& samplesEnd) {
	/*
		           1
		cost  =  -----  *  accumulateForEverySample( || actualOutputs  -  expectedOutputs ||  ^  2 )
               2 * n

		                1
		-->  cost  =  -----  *  accumulateForEverySample( performance( expectedOutputs ) )
		              2 * n
	*/

	flt_t accumulatedPerformances = 0.0;
	for(auto it = samplesBegin; it != samplesEnd; ++it) {
		auto& [inputs, expectedOutputs] = *it;
		updateOutputs<false>(inputs);

		accumulatedPerformances += performance(expectedOutputs);
	}

	return accumulatedPerformances / (2 * std::distance(samplesBegin, samplesEnd));
}

void Network::train(const std::vector<Sample>::const_iterator& samplesBegin, const std::vector<Sample>::const_iterator& samplesEnd, const flt_t eta) {
	resetTotalCostDerivatives(); // this allows to use `+=` on `totalCostDerivative`s

	for(auto it = samplesBegin; it != samplesEnd; ++it) {
		auto& [inputs, expectedOutputs] = *it;

		updateOutputs<true>(inputs);
		resetCurrentCostDerivatives(); // this allows to use `+=` on `currentCostDerivative`s

		std::vector<flt_t> outputDeltas;
		for(size_t y = 0; y != m_nodes.back().size(); ++y) {
			genError(y, m_nodes.back()[y].m_value - expectedOutputs[y]);
			addWeightDerivatives();
			addBiasDerivatives();
		}

		addDerivativesToCostDerivatives();
	}

	scaleAndApplyCostDerivatives(std::distance(samplesBegin, samplesEnd), eta);
}

} /* namespace nn */
