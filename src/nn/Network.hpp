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
#include "Sample.hpp"

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
	 * @brief sets all the derivatives of the parameters to 0, so that += can be used
	 *   To be called before calculating derivatives
	 * @see addDerivativesToCostDerivatives
	 */
	void resetParamCostDerivatives();
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
	 * @brief adds the derivative saved in every parameter (for the currently considered set
	 *   of outputs) to their `costDerivative`
	 */
	void addDerivativesToCostDerivatives();

	/**
	 * @brief scales the `costDerivative` of every parameter by a factor of `1/numberOfSamples`,
	 *   so that it becomes the average of all considered samples.
	 *   Then, using the scaled `costDerivative`, changes the parameters' values by `eta*deriv`
	 * @param numberOfSamples how many samples have been used to calculate the cost derivative
	 * @param eta rate of improvement
	 */
	void scaleAndApplyCostDerivatives(const size_t numberOfSamples, const flt_t eta);

public:
	/**
	 * @brief constructs a fully-connected neural network
	 *   All parameters' values are randomly initialized
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
	 * @brief the cost function, defined as
	 *   sumForEverySample( ||outputs-expectedOutputs||^2 ) / (2*numberOfSamples)
	 * @params [samplesBegin, samplesEnd] the samples containing the expected outputs
	 *   for their inputs
	 * @return the cost of the network
	 */
	flt_t cost(const std::vector<Sample>::const_iterator& samplesBegin, const std::vector<Sample>::const_iterator& samplesEnd);

	/**
	 * @brief trains the network to better perform with the provided samples
	 * @params [samplesBegin, samplesEnd] the samples containing the expected outputs
	 *   for their inputs
	 */
	void train(const std::vector<Sample>::const_iterator& samplesBegin, const std::vector<Sample>::const_iterator& samplesEnd);
};

} /* namespace nn */

#endif /* NETWORK_HPP_ */
