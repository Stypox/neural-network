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

void Network::updateLayer(const size_t l) {
	for(size_t i = 0; i != m_nodes[l].size(); ++i) {
		flt_t sum = std::accumulate(m_nodes[l-1].begin(), m_nodes[l-1].end(), (flt_t)0.0,
				[i](const flt_t& a, const Node& b){
					return a + b.m_outputs[i];
				});
		m_nodes[l][i].setValueFromSum(sum);
	}
}
void Network::updateOutputs(const std::vector<flt_t>& inputs) {
	for(size_t i = 0; i != m_nodes.front().size(); ++i)
		m_nodes[0][i].setValueDirectly(inputs[i]);
	for(size_t i = 1; i != m_nodes.size(); ++i)
		updateLayer(i);
}

flt_t Network::performance(const std::vector<flt_t>& expectedOutputs) {
	flt_t squaresSum = 0;
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		squaresSum += std::pow(m_nodes.back()[y].m_value-expectedOutputs[y], 2);
	}

	return squaresSum;
}

flt_t Network::performanceDerivative(const std::vector<flt_t>& outputDeltas) {
	/*
	  dP                                              dO
	  --  =  accumulateForEveryO( outputDeltas[O]  *  -- )
	  dx                                              dx

	  D is the performance and O is any output node
	*/

	flt_t accumulated = 0;
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		accumulated += outputDeltas[y] * m_nodes.back()[y].m_derivativeSoFar;
	}

	return accumulated;
}
flt_t Network::getDerivative(const size_t nodeAx, const size_t nodeAy, const size_t nodeBy, const std::vector<flt_t>& outputDeltas) {
	size_t nodeBx = nodeAx+1;
	for(size_t y = 0; y != m_nodes[nodeBx].size(); ++y) {
		m_nodes[nodeBx][y].m_derivativeSoFar = 0;
	}
	m_nodes[nodeBx][nodeBy].m_derivativeSoFar = sigDeriv(m_nodes[nodeBx][nodeBy].m_sumInputs) * m_nodes[nodeAx][nodeAy].m_value;

	for(size_t x = nodeBx+1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			/*
			  dB                                                                        dA
			  --  =  sigDeriv(B.m_sumInputs)  *  accumulateForEveryA( A.weightTo(B)  *  -- )
			  dx                                                                        dx

			  A is any node before B --> xB=xA-1
			*/

			flt_t accumulated = 0;
			for(size_t yBef = 0; yBef != m_nodes[x-1].size(); ++yBef) {
				accumulated += m_nodes[x-1][yBef].weightTo(y) * m_nodes[x-1][yBef].m_derivativeSoFar;
			}

			m_nodes[x][y].m_derivativeSoFar = sigDeriv(m_nodes[x][y].m_sumInputs) * accumulated;
		}
	}

	return performanceDerivative(outputDeltas);
}

void Network::genDerivativesForAllWeights(const std::vector<flt_t>& outputDeltas) {
	for(size_t x = 0; x != m_nodes.size()-1; ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			for(size_t yTo = 0; yTo != m_nodes[x+1].size(); ++yTo) {
				m_nodes[x][y].m_weightDerivatives[yTo] = getDerivative(x, y, yTo, outputDeltas);
			}
		}
	}
}
void Network::applyDerivativesToAllWeights(flt_t eta, flt_t maxChange) {
	for(size_t x = 0; x != m_nodes.size()-1; ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			for(size_t yTo = 0; yTo != m_nodes[x].size(); ++yTo) {
				// if the derivative is positive, by decreasing the weight we also decrease the output of the network
				// otherwise the weight should be increased
				flt_t change = m_nodes[x][y].m_weightDerivatives[yTo]*eta;
				if (change > 0) {
					m_nodes[x][y].m_weights[yTo] -= std::min(change, maxChange);
				} else if (change < 0) {
					m_nodes[x][y].m_weights[yTo] -= std::max(change, -maxChange);
				}
			}
		}
	}
}

Network::Network(const std::initializer_list<size_t>& dimensions) {
	for(size_t x = 0; x != dimensions.size()-1; ++x) { // the last neurons have no output
		m_nodes.push_back({});
		for(size_t y = 0; y != dimensions.begin()[x]; ++y) {
			m_nodes.back().push_back(Node{dimensions.begin()[x+1]});
		}
	}
	m_nodes.push_back(std::vector<Node>(*(dimensions.end()-1), Node{0}));
}

std::vector<flt_t> Network::calculate(const std::vector<flt_t>& inputs) {
	updateOutputs(inputs);
	std::vector<flt_t> result;
	for(size_t i = 0; i != m_nodes.back().size(); ++i)
		result.push_back(m_nodes.back()[i].m_value);
	return result;
}

void Network::train(const std::vector<flt_t>& inputs, const std::vector<flt_t>& expectedOutputs) {
	updateOutputs(inputs);

	std::vector<flt_t> outputDeltas;
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		outputDeltas.push_back(m_nodes.back()[y].m_value - expectedOutputs[y]);
	}

	genDerivativesForAllWeights(outputDeltas);
	applyDerivativesToAllWeights(100,0.005);
}

} /* namespace nn */
