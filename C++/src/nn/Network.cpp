#include "Network.hpp"

#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>

using std::pair;
using std::vector;
using vector_iter = vector<nn::Sample>::const_iterator;

namespace nn {

void Network::feedforward(const std::vector<flt_t>& inputs) {
	for(size_t y = 0; y != m_nodes[0].size(); ++y) {
		m_nodes[0][y].a = inputs[y]; // TODO consider putting inputs.at(y) or checking size
	}
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			m_nodes[x][y].z = m_nodes[x][y].bias;
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].z += m_nodes[x-1][yFrom].a * m_nodes[x][y].weights[yFrom];
			}
			m_nodes[x][y].a = m_activationFunction(m_nodes[x][y].z);
		}
	}
}

void Network::momentumSGDMiniBatch(const std::vector<Sample>::const_iterator& samplesBegin,
		const std::vector<Sample>::const_iterator& samplesEnd,
		const flt_t eta,
		const flt_t weightDecayFactor,
		const flt_t momentumCoefficient) {
	// reset
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			m_nodes[x][y].accBiasNabla = 0;
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].accWeightsNabla[yFrom] = 0;
			}
		}
	}

	// accumulate accNablas
	for(auto s = samplesBegin; s != samplesEnd; ++s) {
		backpropagation(*s);
		for(size_t x = 1; x != m_nodes.size(); ++x) {
			for(size_t y = 0; y != m_nodes[x].size(); ++y) {
				m_nodes[x][y].accBiasNabla += m_nodes[x][y].error;
				for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
					m_nodes[x][y].accWeightsNabla[yFrom] += m_nodes[x][y].weightsNabla[yFrom];
				}
			}
		}
	}

	// apply calculated accNablas to velocities
	size_t m = std::distance(samplesBegin, samplesEnd); // mini batch size
	flt_t etaScaled = eta / m;
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			m_nodes[x][y].biasVelocity =
				momentumCoefficient * m_nodes[x][y].biasVelocity -
				etaScaled * m_nodes[x][y].accBiasNabla;
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].weightsVelocity[yFrom] = 
					momentumCoefficient * m_nodes[x][y].weightsVelocity[yFrom] -
					etaScaled * m_nodes[x][y].accWeightsNabla[yFrom];
			}
		}
	}

	// apply calculated velocities to weights
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			m_nodes[x][y].bias += m_nodes[x][y].biasVelocity;
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].weights[yFrom] = 
					weightDecayFactor * m_nodes[x][y].weights[yFrom] +
					m_nodes[x][y].weightsVelocity[yFrom];
			}
		}
	}
}

void Network::backpropagation(const Sample& sample) {
	// feedforward
	feedforward(sample.inputs);

	// backpropagation of output layer
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		m_nodes.back()[y].error = m_costFunction.derivative(m_nodes.back()[y].z, m_nodes.back()[y].a, sample.expectedOutputs[y], m_activationFunction);
		// ^ TODO consider putting sample.expectedOutputs.at(y) or checking size

		for(size_t yFrom = 0; yFrom != m_nodes.end()[-2].size(); ++yFrom) {
			m_nodes.back()[y].weightsNabla[yFrom] = m_nodes.back()[y].error * m_nodes.end()[-2][yFrom].a;
		}
	}

	// backpropagation
	for(size_t x = m_nodes.size()-2; x != 0; --x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			flt_t sd = m_activationFunction.derivative(m_nodes[x][y].z);

			m_nodes[x][y].error = 0;
			for(size_t yTo = 0; yTo != m_nodes[x+1].size(); ++yTo) {
				m_nodes[x][y].error += m_nodes[x+1][yTo].weights[y] * m_nodes[x+1][yTo].error * sd;
			}
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].weightsNabla[yFrom] = m_nodes[x][y].error * m_nodes[x-1][yFrom].a;
			}
		}
	}
}


void Network::momentumSGDEpoch(std::vector<Sample>& trainingSamples,
		const size_t miniBatchSize,
		const flt_t eta,
		const flt_t regularizationParameter,
		const flt_t momentumCoefficient) {
	// reset velocities
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			m_nodes[x][y].biasVelocity = 0;
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].weightsVelocity[yFrom] = 0;
			}
		}
	}
	
	std::random_shuffle(trainingSamples.begin(), trainingSamples.end());
	
	flt_t weightDecayFactor = (1 - eta * regularizationParameter / trainingSamples.size());
	for(size_t start = 0; start < trainingSamples.size(); start += miniBatchSize) {
		auto beg = trainingSamples.begin() + start;
		auto end = std::min(beg + miniBatchSize, trainingSamples.end());

		momentumSGDMiniBatch(beg, end, eta, weightDecayFactor, momentumCoefficient);
	}
}


Network::Network(const std::initializer_list<size_t>& dimensions,
		ActivationFunction& activationFunction,
		CostFunction& costFunction) :
		m_nodes{}, m_activationFunction{activationFunction},
		m_costFunction{costFunction} {
	m_nodes.push_back({});
	for(size_t y = 0; y != dimensions.begin()[0]; ++y) {
		// inputs have no input-connections
		m_nodes.back().push_back(Node{0});
	}

	for(size_t x = 1; x != dimensions.size(); ++x) {
		m_nodes.push_back({});
		for(size_t y = 0; y != dimensions.begin()[x]; ++y) {
			m_nodes.back().push_back(Node{dimensions.begin()[x-1]});
			m_nodes.back().back().bias = random(1);

			flt_t standardDeviation = 1.0 / std::sqrt(dimensions.begin()[x-1]);
			for(size_t yFrom = 0; yFrom != dimensions.begin()[x-1]; ++yFrom) {
				m_nodes.back().back().weights[yFrom] = random(standardDeviation);
			}
		}
	}
}

Network::Network(ActivationFunction& activationFunction, CostFunction& costFunction) :
		m_nodes{}, m_activationFunction{activationFunction},
		m_costFunction{costFunction} {}

std::vector<flt_t> Network::calculate(const std::vector<flt_t>& inputs) {
	feedforward(inputs);
	std::vector<flt_t> result;
	for(size_t i = 0; i != m_nodes.back().size(); ++i)
		result.push_back(m_nodes.back()[i].a);
	return result;
}

void Network::SGD(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta,
		const flt_t regularizationParameter,
		const std::vector<Sample>& testSamples,
		std::ostream& out,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare) {
	momentumSGD(trainingSamples, epochs, miniBatchSize, eta, regularizationParameter, 0.0f, testSamples, out, compare);
}

void Network::momentumSGD(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta,
		const flt_t regularizationParameter,
		const flt_t momentumCoefficient,
		const std::vector<Sample>& testSamples,
		std::ostream& out,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare) {
	out << "Before " << std::setw(std::log10(epochs+1)) << "" <<
		"  -  Accuracy: " << std::setw(std::log10(testSamples.size()) + 1) << evaluate(testSamples, compare) << " / " << testSamples.size() <<
		"  -  Cost: " << cost(testSamples, regularizationParameter) << "\n";
	for(size_t e = 0; e != epochs; ++e) {
		momentumSGDEpoch(trainingSamples, miniBatchSize, eta, regularizationParameter, momentumCoefficient);
		out << "Epoch " << std::setw(std::log10(epochs+1) + 1) << e+1 <<
			"  -  Accuracy: " << std::setw(std::log10(testSamples.size()) + 1) << evaluate(testSamples, compare) << " / " << testSamples.size() <<
			"  -  Cost: " << cost(testSamples, regularizationParameter) << "\n";
	}
}

size_t Network::evaluate(const std::vector<Sample>& testSamples,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare) {
	size_t correct = 0;
	for(auto&& sample : testSamples) {
		std::vector<flt_t> actualOutputs = calculate(sample.inputs);
		correct += compare(sample.expectedOutputs, actualOutputs);
	}
	return correct;
}

flt_t Network::cost(const std::vector<Sample>& samples, const flt_t regularizationParameter) {
	/*
		          1	     |--                                                  regularizationParameter                                        --|
		cost  =  ---  *  |  accumulateForEverySample( m_costFunction() )  +  ------------------------- * accumulateForEveryWeight( weight^2 )  |
                n      |--                                                             2                                                   --|
	*/

	flt_t cost0Acc = 0.0;
	for(auto&& sample : samples) {
		auto& [inputs, expectedOutputs] = sample;
		feedforward(inputs);

		// cost for this set of inputs
		for(size_t y = 0; y != m_nodes.back().size(); ++y) {
			cost0Acc += m_costFunction(m_nodes.back()[y].a, expectedOutputs[y]);
		}
	}

	flt_t weightCostAcc = 0.0;
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(auto&& node : m_nodes[x])
			for(auto&& weight : node.weights)
				weightCostAcc += weight * weight;
	}

	return (cost0Acc + 0.5 * regularizationParameter * weightCostAcc) / samples.size();
}

std::istream& operator>>(std::istream& in, Network& network) {
	size_t xSize;
	in >> xSize;
	network.m_nodes.resize(xSize, std::vector<Node>{});

	// input layer has no parameter
	size_t ySize;
	in >> ySize;
	for(size_t y = 0; y != ySize; ++y) {
		// inputs have no input-connections
		network.m_nodes[0].push_back(Node{0});
	}	

	for(size_t x = 1; x != xSize; ++x) {
		in >> ySize;
		for(size_t y = 0; y != ySize; ++y) {
			network.m_nodes[x].push_back(Node{0});
			in >> network.m_nodes[x].back();
		}
	}

	return in;
}

std::ostream& operator<<(std::ostream& out, const Network& network) {
	out << network.m_nodes.size() << " ";

	// input layer has no parameter
	out << network.m_nodes[0].size() << " ";

	for(size_t x = 1; x != network.m_nodes.size(); ++x) {
		out << network.m_nodes[x].size() << " ";
		for(size_t y = 0; y != network.m_nodes[x].size(); ++y) {
			out << network.m_nodes[x][y];
		}
	}

	return out;
}

} /* namespace nn */
