#ifndef _DEB_HPP_
#define _DEB_HPP_
#include <iostream>
#include <cmath>
#include "nn/utils.hpp"

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

inline void printImage(std::ostream& out, const std::vector<nn::flt_t>& inputs, const std::vector<nn::flt_t>& expectedOutputs) {
	int line = 0;
	for(auto&& pixel : inputs) {
		out << (pixel<0.5 ? "  " : "@@");
		
		++line;
		if(line == (int)std::sqrt(inputs.size())) {
			out << "|\n";
			line = 0;
		}
	}

	out << "-------- " << std::distance(expectedOutputs.begin(), std::max_element(expectedOutputs.begin(), expectedOutputs.end())) << " --------\n";
}

inline void printImage(std::ostream& out, const nn::Sample& image) {
	printImage(out, image.inputs, image.expectedOutputs);
}
#endif
