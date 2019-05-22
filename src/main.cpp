#include "nn/Network.hpp"
#include "deb.hpp"
#include <iomanip>

int main() {
	nn::Network net{5,2,3};

	std::cout << std::fixed << std::setprecision(1);

	std::vector<nn::flt_t> gIn{1.,1.,1.,1.,1.}, gOut{1.,0.,0.},
	                       cIn{0.,0.,1.,1.,1.}, cOut{0.,1.,0.},
	                       fIn{0.,0.,0.,0.,1.}, fOut{0.,0.,1.};
	deb("Before training:");
	deb("Giraffe", net.calculate(gIn), "-", gOut);
	deb("Goat   ", net.calculate(cIn), "-", cOut);
	deb("Ant    ", net.calculate(fIn), "-", fOut);

	for (int i = 0; i < 100000; ++i) {
		net.train(gIn, gOut); // Giraffe
		net.train(cIn, cOut); // Goat
		net.train(fIn, fOut); // Ant
	}

	deb("\nAfter training:");
	deb("Giraffe", net.calculate(gIn), "-", gOut);
	deb("Goat   ", net.calculate(cIn), "-", cOut);
	deb("Ant    ", net.calculate(fIn), "-", fOut);
}
