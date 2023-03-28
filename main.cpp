#include "Network.h"
#include <vector>

/*
Testing file for neural network code.
*/

int main() {
	network N(2, 3, 3, 2, 0.05);
	std::vector<double> in{1, 1};
	N.train({ {1, 1}, {1, 0}, {0, 1}, {0,0}}, {1, 1, 1, 0}, 0.1, 500);
}