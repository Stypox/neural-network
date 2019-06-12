/*
 * shared.hpp
 *
 *  Created on: May 18, 2019
 *      Author: stypox
 */

#ifndef _NN_SHARED_HPP_
#define _NN_SHARED_HPP_

namespace nn {
	using flt_t = float;

	flt_t random();
	flt_t sig(flt_t x);
	flt_t sigDeriv(flt_t x);
}

#endif /* SHARED_HPP_ */
