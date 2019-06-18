#ifndef _NN_NETWORK_HPP_
#define _NN_NETWORK_HPP_

#include <vector>
#include <istream>
#include <ostream>
#include <functional>
#include "utils.hpp"
#include "Node.hpp"
#include "Sample.hpp"

namespace nn {

class Network { public: // TODO
	/*
	   O---------->
	   |         x
	   |
	   |
	   | y
	   v
	*/
	std::vector<std::vector<Node>> m_nodes; // m_nodes[x][y] to access a node

	/**
	 * @brief calculates the value of the output nodes based on the inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 */
	void feedforward(const std::vector<flt_t>& inputs);

	/**
	 * @brief trains the network to better perform with the provided samples using
	 *   the average of the nabla's of all samples
	 * @params [samplesBegin, samplesEnd] the samples containing the expected outputs
	 *   for their inputs
	 * @param eta learning rate
	 */
	void trainMiniBatch(const std::vector<Sample>::const_iterator& samplesBegin,
		const std::vector<Sample>::const_iterator& samplesEnd,
		const flt_t eta);

	/**
	 * @brief calculates the bias' nabla and the weights' nabla of the sample
	 * @param sample the sample containing the expected outputs for the inputs
	 */
	void backpropagation(const Sample& sample);


	/**
	 * @brief the cost function for the current outputs, defined as
	 *   `||outputs-expectedOutputs||^2 / 2`
	 * @param expectedOutputs the expected output for every output node
	 * @return the cost of the network
	 */
	flt_t currentCost(const std::vector<flt_t>& expectedOutputs);

	/**
	 * @brief calculates the derivative of cost function for the considered output node
	 * @param z weighted sum + bias of the considered node's inputs
	 * @param a actual activation of the considered output node
	 * @param y expected activation of the considered output node
	 * @return derivative of cost function
	 */
	flt_t costDerivative(flt_t z, flt_t a, flt_t y);


	/**
	 * @brief applies the stochastic-gradient-descent learning algorithm
	 *   (only for one epoch)
	 * @param trainingSamples the samples to train on, containing the
	 *   expected outputs for their inputs
	 * @param miniBatchSize size of the batch of samples to use for the gradient descent
	 * @param eta learning rate
	 * @see stochasticGradientDescent
	 */
	void stochasticGradientDescentEpoch(std::vector<Sample>& trainingSamples,
		const size_t miniBatchSize,
		const flt_t eta);

public:
	/**
	 * @brief constructs a fully-connected neural network
	 *   All parameters' values are randomly initialized with normal distribution
	 * @param dimensions the length of every layer of nodes
	 */
	Network(const std::initializer_list<size_t>& dimensions);

	/**
	 * @brief Construct an empty Network object
	 * @see operator>>
	 */
	Network() = default;

	/**
	 * @brief calculates the output of the network based on the provided inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 * @return the values of the output nodes
	 */
	std::vector<flt_t> calculate(const std::vector<flt_t>& inputs);

	/**
	 * @brief the cost function, defined as
	 *   `sumForEverySample( ||outputs-expectedOutputs||^2 / 2 ) / numberOfSamples`
	 * @params [samplesBegin, samplesEnd] the samples containing the expected outputs
	 *   for their inputs
	 * @return the cost of the network
	 */
	flt_t cost(const std::vector<Sample>::const_iterator& samplesBegin,
		const std::vector<Sample>::const_iterator& samplesEnd);

	/**
	 * @brief applies the stochastic-gradient-descent learning algorithm
	 * @param trainingSamples the samples to train on, containing the
	 *   expected outputs for their inputs
	 * @param epochs number of epochs
	 * @param miniBatchSize size of the batch of samples to use for the gradient descent
	 * @param eta learning rate
	 * @see stochasticGradientDescentEpoch
	 */
	void stochasticGradientDescent(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta);

	/**
	 * @brief applies the stochastic-gradient-descent learning algorithm,
	 *   while also printing network statistics after every epoch
	 * @param trainingSamples the samples to train on, containing the
	 *   expected outputs for their inputs
	 * @param epochs number of epochs
	 * @param miniBatchSize size of the batch of samples to use for the gradient descent
	 * @param eta learning rate
	 * @param testSamples the samples to use for testing, containing the
	 *   expected outputs for their inputs
	 * @param out output stream on which to print network statistics
	 * @param compare function that compares the actual outputs and the expected outputs
	 *   and returns `true` if they somehow match, otherwise `false`
	 * @see stochasticGradientDescentEpoch
	 * @see evaluate
	 */
	void stochasticGradientDescent(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta,
		const std::vector<Sample>& testSamples,
		std::ostream& out,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare);
	
	/**
	 * @brief calculates how many test samples are correctly recognized by the network
	 * @param testSamples the samples to use for testing, containing the
	 *   expected outputs for their inputs
	 * @param compare function that compares the actual outputs and the expected outputs
	 *   and returns `true` if they somehow match, otherwise `false`
	 * @return the count of test samples that the network recognises correctly
	 */
	size_t evaluate(const std::vector<Sample>& testSamples,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare);
	
	/**
	 * @brief read network parameters from an input stream
	 * @param in input stream
	 * @param network the network to save the parameters in
	 * @return in
	 */
	friend std::istream& operator>>(std::istream& in, Network& network);

	/**
	 * @brief write network parameters to an output stream
	 * @param out output stream
	 * @param network the network to write
	 * @return out
	 */
	friend std::ostream& operator<<(std::ostream& out, const Network& network);
};

} /* namespace nn */

#endif /* _NN_NETWORK_HPP_ */
