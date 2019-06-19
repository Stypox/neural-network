#ifndef _NN_COSTFUNCTION_HPP_
#define _NN_COSTFUNCTION_HPP_

#include "utils.hpp"
#include <cmath>

namespace nn {

/**
 * @brief cost function and cost derivative
 * @param z weighted sum + bias of the considered output node's inputs
 * @param a actual activation of the considered output node
 * @param y expected activation of the considered output node
 */
class CostFunction {
public:
	virtual flt_t operator()(const flt_t a, const flt_t y) const = 0;
	virtual flt_t derivative(const flt_t z, const flt_t a, const flt_t y) const = 0;
};

class QuadraticCost : public CostFunction {
public:
	flt_t operator()(const flt_t a, const flt_t y) const final {
		return 0.5 * (a-y)*(a-y);
	}
	flt_t derivative(const flt_t z, const flt_t a, const flt_t y) const final {
		return (a-y) * sigDeriv(z);
	}
};
inline QuadraticCost quadraticCost;

class CrossEntropyCost : public CostFunction {
	constexpr flt_t customLog(const flt_t a) const { // prevent log(0)
		if (a==0) return std::numeric_limits<flt_t>::min();
		return std::log(a);
	}
public:
	flt_t operator()(const flt_t a, const flt_t y) const final {
		return - y*customLog(a) - (1-y)*customLog(1-a);
	}
	flt_t derivative(const flt_t, const flt_t a, const flt_t y) const final {
		return a-y;
	}
};
inline CrossEntropyCost crossEntropyCost;

} // namespace nn

#endif // _NN_COSTFUNCTION_HPP_