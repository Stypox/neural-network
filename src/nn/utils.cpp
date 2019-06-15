#include "utils.hpp"
#include <random>
#include <cmath>

namespace nn {

	flt_t random() {
		static std::random_device rd;
		static std::mt19937 engine{rd()};
		static std::normal_distribution<> dist;
		return dist(engine);
	}
	flt_t sig(flt_t x) {
		if (x > 0)
			return (flt_t)1.0 / ((flt_t)1.0 + std::exp(-x));
		else
			return (flt_t)1.0 - (flt_t)1.0 / ((flt_t)1.0 + std::exp(x));
	}
	flt_t sigDeriv(flt_t x) {
		const flt_t exp = std::exp(-std::abs(x));
		return exp / std::pow(1+exp, 2);
	}

}
