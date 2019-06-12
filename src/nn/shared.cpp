/*
 * shared.cpp
 *
 *  Created on: May 18, 2019
 *      Author: stypox
 */

#include "shared.hpp"
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
		return (flt_t)1.0 / ((flt_t)1.0 + std::exp(-x));
	}

	flt_t sigDeriv(flt_t x) {
		const flt_t exp = std::exp(-x);
		return exp / std::pow(1+exp, 2);
	}
}
