#include "Network.h"
#include <vector>

int main() {
	network N(2, 3, 3, 2, 0.05);
	std::vector<double> in{1, 1};
	N.train({1, 1}, 1, 0.5);
	N.train({ 1, 0 }, 0, 0.5);
	N.train({ 0, 1 }, 0, 0.5);
	N.train({ 0, 0 }, 0, 0.5);
}