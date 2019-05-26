/*
 * Network.hpp
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
	 * @param x layer to update, must be >= 1 (not the inputs layer)
	 * @tparam training prepare training data in nodes
	 */
	template<bool training>
	void updateLayer(const size_t x);
	/**
	 * @brief updates the output layer based on the inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 */
	template<bool training>
	void updateOutputs(const std::vector<flt_t>& inputs);

	/**
	 * @brief calculates the performance of the network based on the info saved in the output nodes
	 * @param expectedOutputs expected outputs
	 * @return vectorial distance squared (^2) between the current outputs and the excpected ones
	 */
	flt_t performance(const std::vector<flt_t>& expectedOutputs);

	/**
	 * @brief sets all the derivatives of the parameters to 0, so that += can be used
	 *   To be called before calculating derivatives
	 * @see addWeightDerivatives
	 * @see addBiasDerivatives
	 */
	void resetParamDerivatives();
	/**
	 * @brief sets all derivativeFromHereOn of the nodes to 0, so that += can be used
	 *   To be called before generating the derivativeFromHereOn for every output
	 * @see genDerivativesFromHereOn
	 */
	void resetDerivativesFromHereOn();

	/**
	 * @brief calculates and saves the derivativefromHereOn in each node
	 * @param consideredOutput y of the considered output node
	 * @param outputDelta (actualValue-expectedValue) of the considered output
	 */
	void genDerivativesFromHereOn(const size_t consideredOutput, const flt_t outputDelta);
	/**
	 * @brief using the derivativefromHereOn saved in the nodes, calculates the derivative of
	 *   the currently considered output over every weight and adds it to the weight's derivative
	 */
	void addWeightDerivatives();
	/**
	 * @brief using the derivativefromHereOn saved in the nodes, calculates the derivative of
	 *   the currently considered output over every bias and adds it to the bias' derivative
	 */
	void addBiasDerivatives();

	/**
	 * @brief using the derivatives saved in each parameter, changes their values by
	 *   (deriv>0 ? min(eta*deriv, maxChange) : max(eta*deriv, -maxChange))
	 * @param eta rate of improvement
	 * @param maxChange maximum change to apply to the weights
	 */
	void applyAllDerivatives(const flt_t eta, const flt_t maxChange);

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
	 * @param expectedOutputs array of expected values for every output node
	 */
	void train(const std::vector<flt_t>& inputs, const std::vector<flt_t>& expectedOutputs);
};

} /* namespace nn */

#endif /* NETWORK_HPP_ */
