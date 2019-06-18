#include "Network.hpp"

#include <numeric>
#include <cmath>
#include <algorithm>

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
			m_nodes[x][y].a = sig(m_nodes[x][y].z);
		}
	}
}

void Network::trainMiniBatch(const std::vector<Sample>::const_iterator& samplesBegin,
	const std::vector<Sample>::const_iterator& samplesEnd,
	const flt_t eta) {
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

	// apply calculated accNablas
	flt_t ratio = eta / std::distance(samplesBegin, samplesEnd);
	for(size_t x = 1; x != m_nodes.size(); ++x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			m_nodes[x][y].bias -= m_nodes[x][y].accBiasNabla * ratio;
			for(size_t yFrom = 0; yFrom != m_nodes[x-1].size(); ++yFrom) {
				m_nodes[x][y].weights[yFrom] -= m_nodes[x][y].accWeightsNabla[yFrom] * ratio;
			}
		}
	}
}

void Network::backpropagation(const Sample& sample) {
	// feedforward
	feedforward(sample.inputs);

	// backpropagation of output layer
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		m_nodes.back()[y].error = m_costDerivative(m_nodes.back()[y].z, m_nodes.back()[y].a, sample.expectedOutputs[y]);
		// ^ TODO consider putting sample.expectedOutputs.at(y) or checking size

		for(size_t yFrom = 0; yFrom != m_nodes.end()[-2].size(); ++yFrom) {
			m_nodes.back()[y].weightsNabla[yFrom] = m_nodes.back()[y].error * m_nodes.end()[-2][yFrom].a;
		}
	}

	// backpropagation
	for(size_t x = m_nodes.size()-2; x != 0; --x) {
		for(size_t y = 0; y != m_nodes[x].size(); ++y) {
			flt_t sd = sigDeriv(m_nodes[x][y].z);

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

flt_t Network::currentCost(const std::vector<flt_t>& expectedOutputs) {
	flt_t squaresSum = 0;
	for(size_t y = 0; y != m_nodes.back().size(); ++y) {
		squaresSum += std::pow(m_nodes.back()[y].a-expectedOutputs[y], 2);
	}

	return squaresSum / 2;
}


void Network::stochasticGradientDescentEpoch(std::vector<Sample>& trainingSamples,
		const size_t miniBatchSize,
		const flt_t eta) {
	std::random_shuffle(trainingSamples.begin(), trainingSamples.end());
	
	for(size_t start = 0; start < trainingSamples.size(); start += miniBatchSize) {
		auto beg = trainingSamples.begin() + start;
		auto end = std::min(beg + miniBatchSize, trainingSamples.end());

		trainMiniBatch(beg, end, eta);
	}
}


Network::Network(const std::initializer_list<size_t>& dimensions,
		flt_t(*costDerivative)(flt_t, flt_t, flt_t)) :
		m_nodes{}, m_costDerivative{costDerivative} {
	m_nodes.push_back({});
	for(size_t y = 0; y != dimensions.begin()[0]; ++y) {
		// inputs have no input-connections
		m_nodes.back().push_back(Node{0});
	}

	for(size_t x = 1; x != dimensions.size(); ++x) {
		m_nodes.push_back({});
		for(size_t y = 0; y != dimensions.begin()[x]; ++y) {
			m_nodes.back().push_back(Node{dimensions.begin()[x-1]});
			m_nodes.back().back().bias = random();
			for(size_t yFrom = 0; yFrom != dimensions.begin()[x-1]; ++yFrom) {
				m_nodes.back().back().weights[yFrom] = random();
			}
		}
	}
}

Network::Network(flt_t(*costDerivative)(flt_t, flt_t, flt_t)) :
		m_nodes{}, m_costDerivative{costDerivative} {}

std::vector<flt_t> Network::calculate(const std::vector<flt_t>& inputs) {
	feedforward(inputs);
	std::vector<flt_t> result;
	for(size_t i = 0; i != m_nodes.back().size(); ++i)
		result.push_back(m_nodes.back()[i].a);
	return result;
}

void Network::stochasticGradientDescent(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta) {
	for(size_t e = 0; e != epochs; ++e) {
		stochasticGradientDescentEpoch(trainingSamples, miniBatchSize, eta);
	}
}

void Network::stochasticGradientDescent(std::vector<Sample> trainingSamples,
		const size_t epochs,
		const size_t miniBatchSize,
		const flt_t eta,
		const std::vector<Sample>& testSamples,
		std::ostream& out,
		std::function<bool(const std::vector<flt_t>&, const std::vector<flt_t>&)> compare) {
	for(size_t e = 0; e != epochs; ++e) {
		stochasticGradientDescentEpoch(trainingSamples, miniBatchSize, eta);
		out << "Epoch " << e+1 << ": " << evaluate(testSamples, compare) << " / " << testSamples.size() << "\n";
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
		feedforward(inputs);

		accumulatedPerformances += currentCost(expectedOutputs);
	}

	return accumulatedPerformances / std::distance(samplesBegin, samplesEnd);
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
