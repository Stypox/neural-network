#include "nn/Network.hpp"
#include "deb.hpp"
#include <iomanip>

int main() {
	nn::Network net{5,5,4};

	std::cout << std::fixed << std::setprecision(1);

	std::vector<nn::flt_t>
		giIn{1.,1.,0.,0.,1.}, giOut{1.,0.,0.,0.},
		elIn{1.,1.,1.,1.,1.}, elOut{0.,1.,0.,0.},
		goIn{0.,0.,1.,1.,1.}, goOut{0.,0.,1.,0.},
		anIn{0.,0.,0.,0.,1.}, anOut{0.,0.,0.,1.};

	deb("Before training:");
	deb("Giraffe ", net.calculate(giIn), "-", giOut);
	deb("Elephant", net.calculate(elIn), "-", elOut);
	deb("Goat    ", net.calculate(goIn), "-", goOut);
	deb("Ant     ", net.calculate(anIn), "-", anOut);

	for (int i = 0; i < 50000; ++i) {
		net.train(giIn, giOut); // Giraffe
		net.train(elIn, elOut); // Elephant
		net.train(goIn, goOut); // Goat
		net.train(anIn, anOut); // Ant
	}

	deb("\nAfter training:");
	deb("Giraffe ", net.calculate(giIn), "-", giOut);
	deb("Elephant", net.calculate(elIn), "-", elOut);
	deb("Goat    ", net.calculate(goIn), "-", goOut);
	deb("Ant     ", net.calculate(anIn), "-", anOut);
}
