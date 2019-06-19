#include "utils.hpp"
#include <random>
#include <cmath>

namespace nn {

	flt_t random(const flt_t standardDeviation) {
		static std::random_device rd;
		static std::mt19937 engine{rd()};
		return std::normal_distribution{(flt_t)0.0, standardDeviation}(engine);
	}

}
