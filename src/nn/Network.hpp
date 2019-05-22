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

	/**
	 * @param expectedOutputs expected outputs
	 * @return vectorial distance between the current outputs and the excpected ones
	 */
	flt_t outputDistance(const std::vector<flt_t>& expectedOutputs);

	/**
	 * @brief calculates the derivative d(outputDistance)/d(weight) based on the info saved in the output nodes
	 * @param outputMultiplier 1/outputDistance
	 * @param outputDeltas array of |outputs[i] - expectedOutputs[i]| for every i
	 * @return d(output)/d(weight)
	 */
	flt_t outputDistanceDerivative(const flt_t outputMultiplier, const std::vector<flt_t>& outputDeltas);
	/**
	 * @brief calculates the derivative do/dw, where:
	 *   - w is the weight of the connection between nodeA(nodeAx, nodeAy) and nodeB(nodeAx+1, nodeBy)
	 *   - o is the output
	 * @param nodeAx see above
	 * @param nodeAy see above
	 * @param nodeBy see above; nodeBx is not needed since it is always =nodeAx+1
	 * @param outputMultiplier 1/outputDistance
	 * @param outputDeltas array of |outputs[i] - expectedOutputs[i]| for every i
	 * @return derivative d(outputDistance)/d(weight)
	 */
	flt_t getDerivative(const size_t nodeAx, const size_t nodeAy, const size_t nodeBy, const flt_t outputMultiplier, const std::vector<flt_t>& outputDeltas);

	/**
	 * @brief generates the derivative for each connection and saved them inside the nodes
	 * @param outputMultiplier 1/outputDistance
	 * @param outputDeltas array of |outputs[i] - expectedOutputs[i]| for every i
	 */
	void genDerivativesForAllWeights(const flt_t outputMultiplier, const std::vector<flt_t>& outputDeltas);
	/**
	 * @brief using the results of the derivatives, changes the weights by
	 *   (deriv>0 ? min(eta*deriv, maxChange) : max(eta*deriv, -maxChange))
	 * @param eta rate of improvement
	 * @param maxChange maximum change to apply to the weights
	 */
	void applyDerivativesToAllWeights(flt_t eta, flt_t maxChange);

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

	/**
	 * @brief trains the network to better perform with the provided inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 * @param expectedOutputs array of expected values for every node in the last layer of the network
	 * @return outputDistance before training
	 */
	flt_t train(const std::vector<flt_t>& inputs, const std::vector<flt_t>& expectedOutputs);
};

} /* namespace nn */

#endif /* NETWORK_HPP_ */
