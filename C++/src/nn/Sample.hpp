#ifndef _NN_SAMPLE_HPP_
#define _NN_SAMPLE_HPP_

#include <vector>
#include "utils.hpp"

namespace nn {

class Sample {
	std::vector<flt_t> inputs;
	std::vector<flt_t> expectedOutputs;

public:
	/**
	 * @brief Construct a Sample to be used for autoclassifiers:
	 * inputs will also be used as expected outputs
	 *
	 * @param data the data to use both as inputs and expected outputs
	 */
	Sample(const std::vector<flt_t>& data)
			: inputs{data}, expectedOutputs{} {}
	/**
	 * @brief Construct a Sample to be used for autoclassifiers:
	 * inputs will also be used as expected outputs
	 *
	 * @param data the data to use both as inputs and expected outputs
	 */
	Sample(const std::vector<flt_t>&& data)
			: inputs{data}, expectedOutputs{} {}

	/**
	 * @brief Construct a Sample with inputs and expected outputs
	 *
	 * @param inputs the inputs
	 * @param expectedOutputs the expected outputs corresponding to the inputs
	 */
	Sample(const std::vector<flt_t>& inputs, const std::vector<flt_t>& expectedOutputs)
			: inputs{inputs}, expectedOutputs{expectedOutputs} {}

	/**
	 * @brief Construct a Sample with inputs and the classifer output
	 *
	 * @param inputs the inputs
	 * @param expectedClass the expected class corresponding to the inputs
	 * @param classCount the number of all possible classes, which determines the expected outputs length
	 */
	Sample(const std::vector<flt_t>& inputs, const size_t expectedClass, const size_t classCount)
			: inputs{inputs}, expectedOutputs(classCount, (flt_t) 0.0) {
		expectedOutputs[expectedClass] = (flt_t) 1.0;
	}

	const std::vector<flt_t>& getInputs() const {
		return inputs;
	}

	const std::vector<flt_t>& getExpectedOutputs() const {
		return expectedOutputs.size() == 0 ? inputs : expectedOutputs;
	}

	void swap(nn::Sample& other) {
		std::swap(inputs, other.inputs);
		std::swap(expectedOutputs, other.expectedOutputs);
	}
};

} // namespace nn

#endif // _NN_SAMPLE_HPP_