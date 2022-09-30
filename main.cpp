#include "Network.h"
#include <vector>

int main() {
	network N(2, 3, 3, 2);
	std::vector<double> in{ 0.5, 0.5 };
	N.test(in);
}