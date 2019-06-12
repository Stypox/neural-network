#ifndef _NN_SAMPLE_HPP_
#define _NN_SAMPLE_HPP_

#include <vector>
#include "shared.hpp"

namespace nn {
	struct Sample {
		std::vector<flt_t> inputs;
		std::vector<flt_t> expectedOutputs;
	};
} // namespace nn

#endif // _NN_SAMPLE_HPP_