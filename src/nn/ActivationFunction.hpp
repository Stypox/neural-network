#ifndef _NN_ACTIVATIONFUNCTION_HPP_
#define _NN_ACTIVATIONFUNCTION_HPP_

#include "utils.hpp"
#include <cmath>

namespace nn {

class ActivationFunction {
public:
	virtual flt_t operator()(const flt_t z) const = 0;
	virtual flt_t derivative(const flt_t z) const = 0;
};

class Sigmoid : public ActivationFunction {
	flt_t operator()(const flt_t z) const final {
		if (z > 0)
			return 1.0 / (1.0 + std::exp(-z));
		else
			return 1.0 - 1.0 / (1.0 + std::exp(z));
	}
	flt_t derivative(const flt_t z) const final {
		const flt_t exp = std::exp(-std::abs(z));
		return exp / std::pow(1+exp, 2);
	}
};
inline Sigmoid sigmoid;

class FastSigmoid : public ActivationFunction {
	flt_t operator()(const flt_t z) const final {
		return z / (1.0 + std::abs(z));
	}
	flt_t derivative(const flt_t z) const final {
		const flt_t denom = std::abs(z) + 1;
		return 1.0 / (denom * denom);
	}
};
inline FastSigmoid fastSigmoid;

class Tanh : public ActivationFunction {
	flt_t operator()(const flt_t z) const final {
		return std::tanh(z);
	}
	flt_t derivative(const flt_t z) const final {
		const flt_t res = 1.0 / std::cosh(z);
		return res * res;
	}
};
inline Tanh tanh;

} // namespace nn

#endif // _NN_ACTIVATIONFUNCTION_HPP_