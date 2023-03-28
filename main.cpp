#include "Network.h"
#include <vector>

/*
Testing file for neural network code.

Sample test is using an OR gate.
*/

int main() {
	network N(2, 3, 3, 2, 0.05);
	N.train({ {1, 1}, {1, 0}, {0, 1}, {0,0}}, {1, 1, 1, 0}, 0.1, 500);
}