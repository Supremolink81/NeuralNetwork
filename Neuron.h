#ifndef NEURON
#define NEURON
#endif

#include <random>
#include <vector>

class neuron {

	private:
		double value, bias;
		std::vector<double> weights;

	public:
		//Default constructor
		neuron() {
			std::default_random_engine generator{ static_cast<unsigned long int>(time(0)) };
			std::uniform_real_distribution<double> distribution(-5.0, 5.0);
			value = 0.0;
			bias = distribution(generator);
		}

		neuron(size_t w) {
			std::default_random_engine generator{static_cast<unsigned long int>(time(0))};
			std::uniform_real_distribution<double> distribution(-5.0, 5.0);
			value = 0.0;
			bias = distribution(generator);
			weights.resize(w);
			for (int i = 0; i < w; i++) weights[i] = distribution(generator);
		}

		//Value constructor
		neuron(size_t w, double v) {
			std::default_random_engine generator{ static_cast<unsigned long int>(time(0)) };
			std::uniform_real_distribution<double> distribution(-5.0, 5.0);
			value = v;
			bias = distribution(generator);
			weights.resize(w);
			for (int i = 0; i < w; i++) weights[i] = distribution(generator);
		}

		/// <summary>
		/// Returns this node's value.
		/// </summary>
		double get_value() {
			return value;
		}

		/// <summary>
		/// Returns this node's bias.
		/// </summary>
		double get_bias() {
			return bias;
		}

		/// <summary>
		/// Returns the weight of this node's connection to a given node.
		/// </summary>
		double get_weight(size_t index) {
			return weights[index];
		}

		/// <summary>
		/// Sets this node's value equal to the parameter.
		/// </summary>
		void set_value(double v) {
			value = v;
		}

};