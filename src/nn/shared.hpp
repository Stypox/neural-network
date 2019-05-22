/*
 * shared.hpp
 *
 *  Created on: May 18, 2019
 *      Author: stypox
 */

#ifndef SHARED_HPP_
#define SHARED_HPP_

namespace nn {
	using flt_t = float;

	flt_t random();
	flt_t sig(flt_t x);
	flt_t sigDeriv(flt_t x);
}

#endif /* SHARED_HPP_ */
