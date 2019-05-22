/*
 * Network.h
 *
 *  Created on: May 17, 2019
 *      Author: stypox
 */

#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include <ostream>
#include "shared.hpp"
#include "Node.hpp"

namespace nn {

class Network {
	/*
	   O---------->
	   |         x
	   |
	   |
	   | y
	   v
	*/
	std::vector<std::vector<Node>> m_nodes;

	/**
	 * @brief updates layer (to be called after the inputs have changed)
	 * @param l layer to update, must be >= 1 (not the inputs layer)
	 */
	void updateLayer(const size_t l);
	/**
	 * @brief updates the output layer based on the inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 */
	void updateOutputs(const std::vector<flt_t>& inputs);

public:
	/**
	 * @brief constructs a fully-connected neural network
	 *   All the weights are randomly initialized
	 * @param dimensions the length of every layer of nodes
	 */
	Network(const std::initializer_list<size_t>& dimensions);

	/**
	 * @brief calculates the output of the network based on the provided inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 * @return the values of the output nodes
	 */
	std::vector<flt_t> calculate(const std::vector<flt_t>& inputs);
};

} /* namespace nn */

#endif /* NETWORK_HPP_ */
