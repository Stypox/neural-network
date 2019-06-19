#ifndef _NN_NETWORK_HPP_
#define _NN_NETWORK_HPP_

#include <vector>
#include <istream>
#include <ostream>
#include <functional>
#include "utils.hpp"
#include "Node.hpp"
#include "Sample.hpp"
#include "CostFunction.hpp"

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

	ActivationFunction& m_activationFunction;
	CostFunction& m_costFunction;

	/**
	 * @brief calculates the value of the output nodes based on the inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 */
	void feedforward(const std::vector<flt_t>& inputs);

	/**
	 * @brief trains the network to better perform with the provided samples using
	 *   the average of the nabla's of all samples and the "velocity" of every node
	 * @params [samplesBegin, samplesEnd] the samples containing the expected outputs
	 *   for their inputs
	 * @param eta learning rate
	 * @param weightDecayFactor `1 - eta * regularizationParameter / n` where `n` is the
	 *   number of all training samples (not the size of the mini batch)
	 * @param momentumCoefficient factor to scale the "velocity" of the parameter by,
	 *   every iteration. Set to 0 to run exactly as standard stochastic-gradient-descent.
	 */
	void momentumSGDMiniBatch(const std::vector<Sample>::const_iterator& samplesBegin,
		const std::vector<Sample>::const_iterator& samplesEnd,
		const flt_t eta,
		const flt_t weightDecayFactor,
		const flt_t momentumCoefficient);

	/**
	 * @brief calculates the bias' nabla and the weights' nabla of the sample
	 * @param sample the sample containing the expected outputs for the inputs
	 */
	void backpropagation(const Sample& sample);


	/**
	 * @brief applies the momentum-based stochastic-gradient-descent learning algorithm
	 *   (only for one epoch)
	 * @param trainingSamples the samples to train on, containing the
	 *   expected outputs for their inputs
	 * @param miniBatchSize size of the batch of samples to use for the gradient descent
	 * @param eta learning rate
	 * @param regularizationParameter how much the weights should be prevented from
	 *   becoming big. Set to 0 if no regularization is wanted.
	 * @param momentumCoefficient factor to scale the "velocity" of the parameter by,
	 *   every iteration. Set to 0 to run exactly as standard stochastic-gradient-descent.
	 * @see stochasticGradientDescent
	 */
	void momentumSGDEpoch(std::vector<Sample>& trainingSamples,
		const size_t miniBatchSize,
		const flt_t eta,
		const flt_t regularizationParameter,
		const flt_t momentumCoefficient);

public:
	/**
	 * @brief constructs a fully-connected neural network
	 *   All parameters' values are randomly initialized with normal distribution
	 * @param dimensions the length of every layer of nodes
	 * @param activationFunction @see nn::ActivationFunction class
	 * @param costFunction @see nn::CostFunction class
	 */
	Network(const std::initializer_list<size_t>& dimensions,
		ActivationFunction& activationFunction,
		CostFunction& costFunction);

	/**
	 * @brief constructs an empty neural network
	 * @param activationFunction @see nn::ActivationFunction class
	 * @param costFunction @see nn::CostFunction class
	 * @see operator>>
	 */
	Network(ActivationFunction& activationFunction, CostFunction& costFunction);

	/**
	 * @brief calculates the output of the network based on the provided inputs
	 * @param inputs array of inputs of the same length as the first layer of the network
	 * @return the values of the output nodes
	 */
	std::vector<flt_t> calculate(const std::vector<flt_t>& inputs);

	/**
	 * @brief the cost function over all samples and weights
	 * @param samples the samples on which to calculate the cost
	 * @param regularizationParameter how much the weights should be prevented from
	 *   becoming big. Set to 0 if no regularization is wanted. This is typical of
	 *   training but takes part in the cost.
	 * @return cost
	 */
	flt_t cost(const std::vector<Sample>& samples, const flt_t regularizationParameter);

	/**
	 * @brief applies the stochastic-gradient-descent learning algorithm,
	 *   while also printing network statistics after every epoch
	 * @param trainingSamples the samples to train on, containing the
	 *   expected outputs for their inputs
	 * @param epochs number of epochs
	 * @param miniBatchSize size of the batch of samples to use for the gradient descent
	 * @param eta learning rate
	 * @param regularizationParameter how much the weights should be prevented from
	 *   becoming big. Set to 0 if no regularization is wanted.
	 * @param testSamples the samples to use for testing, containing the
	 *   expected outputs for their inputs
	 * @param out output stream on which to print network statistics
	 * @param compare function that compares the actual outputs and the expected outputs
	 *   and returns `true` if they somehow match, otherwise `false`
	 * @see stochasticGradientDescentEpoch
	 * @see evaluate
	 */
	void SGD(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta,
		const flt_t regularizationParameter,
		const std::vector<Sample>& testSamples,
		std::ostream& out,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare);

	/**
	 * @brief applies the momentum-based stochastic-gradient-descent learning algorithm,
	 *   while also printing network statistics after every epoch
	 * @param trainingSamples the samples to train on, containing the
	 *   expected outputs for their inputs
	 * @param epochs number of epochs
	 * @param miniBatchSize size of the batch of samples to use for the gradient descent
	 * @param eta learning rate
	 * @param regularizationParameter how much the weights should be prevented from
	 *   becoming big. Set to 0 if no regularization is wanted.
	 * @param momentumCoefficient factor to scale the "velocity" of the parameter by,
	 *   every iteration. Set to 0 to run exactly as standard stochastic-gradient-descent.
	 * @param testSamples the samples to use for testing, containing the
	 *   expected outputs for their inputs
	 * @param out output stream on which to print network statistics
	 * @param compare function that compares the actual outputs and the expected outputs
	 *   and returns `true` if they somehow match, otherwise `false`
	 * @see stochasticGradientDescentEpoch
	 * @see evaluate
	 */
	void momentumSGD(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta,
		const flt_t regularizationParameter,
		const flt_t momentumCoefficient,
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
