#ifndef _NN_UTILS_HPP_
#define _NN_UTILS_HPP_

namespace nn {
	using flt_t = float;

	flt_t random();
	flt_t sig(flt_t x);
	flt_t sigDeriv(flt_t x);
}

#endif /* _NN_UTILS_HPP_ */
