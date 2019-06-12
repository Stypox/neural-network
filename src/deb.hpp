#ifndef _DEB_HPP_
#define _DEB_HPP_
#include <iostream>
#include "nn/shared.hpp"

// TODO remove this file, useful only for debugging

inline void deb() {std::cout<<"\n";}
template<class T, class... Ts> void deb(T t, Ts... args) {
	if constexpr(std::is_same_v<T,std::vector<nn::flt_t>>) {
		for(auto&& e : t) {
			std::cout << e << " ";
		}
	}
	else {
		std::cout<<t<<" ";
	}
	deb(args...);
}
#endif
