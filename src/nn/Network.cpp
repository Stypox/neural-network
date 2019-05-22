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

} /* namespace nn */
