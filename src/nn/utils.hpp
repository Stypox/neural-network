#ifndef _NN_UTILS_HPP_
#define _NN_UTILS_HPP_

namespace nn {
	using flt_t = float;

	flt_t random(const flt_t standardDeviation);
	flt_t sig(const flt_t x); // consider renaming
	flt_t sigDeriv(const flt_t x);
}

#endif /* _NN_UTILS_HPP_ */
