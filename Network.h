#ifndef NETWORK
#define NETWORK
#endif

#include "Neuron.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
using namespace Eigen;

class network {

    private:
        std::vector<neuron> input_layer, output_layer;
        std::vector<std::vector<neuron>> internal_layers;

        /// <summary>
        /// Activation function to make the input between 0 and 1.
        /// </summary>
        /// <param name="val">The parameter for the activation function</param>
        inline double sigmoid(double val) {
            return 1.0 / (1.0 + std::exp(-val));
        }

        /// <summary>
        /// Determines the cost of a single test in the neural network.
        /// </summary>
        inline double cost() {
            double total_cost = 0;
            for (size_t i = 0; i < output_layer.size(); i++) {
                double cost = abs(output_layer[i].get_value() - 1.0);
                total_cost += cost * cost;
            }
            return total_cost;
        }

        MatrixXd initialize_weight_matrix(std::vector<neuron>& input, std::vector<neuron>& output) {
            MatrixXd weight_matrix(output.size(), input.size());
            for (size_t row = 0; row < output.size(); row++) {
                for (size_t col = 0; col < input.size(); col++) {
                    weight_matrix(row, col) = input[col].get_weight(row);
                }
            }
            return weight_matrix;
        }

        /// <summary>
        /// Returns a vector of values/biases from a layer of neurons.
        /// </summary>
        /// <param name="input">: The layer of neurons to get the values from</param>
        /// <param name="biases">: Whether or not this should be a bias vector (defaults to false)</param>
        VectorXd initialize_vector(std::vector<neuron>& input, bool biases = false) {
            VectorXd vec(input.size());
            for (size_t i = 0; i < input.size(); i++) {
                if (biases) vec(i) = input[i].get_bias();
                else vec(i) = input[i].get_value();
            }
            return vec;
        }

        void get_results(std::vector<neuron>& output, MatrixXd& weights, VectorXd& values, VectorXd& biases) {
            VectorXd results_vec = (weights * values) + biases;
            for (size_t i = 0; i < output.size(); i++) output[i].set_value(sigmoid(results_vec[i]));
        }

    public:

        /// <summary>
        /// Initializes a new neural network, given sizes for each layer.
        /// </summary>
        /// <param name="input_size">: size of input layer</param>
        /// <param name="layers">: number of internal layers in the neural network</param>
        /// <param name="layer_size">: size of each internal layer</param>
        /// <param name="output_size">: size of output layer</param>
        network(size_t input_size, size_t layers, size_t layer_size, size_t output_size) {
            assert(input_size > 0 && layers > 0 && layer_size > 0 && output_size > 0, "all input sizes must be at least 1");

            //Initialize input layer
            input_layer.resize(input_size);
            for (neuron& n : input_layer) n = neuron(layer_size);

            //Initialize each internal layer (excluding last layer)
            internal_layers.resize(layers);
            for (size_t i = 0; i < layers; i++) {
                internal_layers[i].resize(layer_size);
                if (i < layers - 1) for (neuron& n : internal_layers[i]) n = neuron(layer_size);
            }

            //Last internal layer's weights depend on the side of the output layer
            for (neuron& n : internal_layers[layers - 1]) n = neuron(output_size);
            output_layer.resize(output_size);
        }

        /// <summary>
        /// Sets the values of the output layer, depending on the inputs given.
        /// 
        /// This function will initialize the values of each input neuron with the values
        /// in the input list, then set the values of all layers in the network based on the weights, 
        /// values, and biases of the neurons. Linear algebra is used to optimize operations, as 
        /// the library used supports SIMD (Single Instruction, Multiple Data) to perform multiple
        /// computations at the same time.
        /// 
        /// At the end of this function, the output layer's neurons will all have their 
        /// activations set to various numbers between 0 and 1.
        /// 
        /// The amount of inputs must be equal to the input layer's size.
        /// </summary>
        /// <param name="inputs"></param>
        void test(std::vector<double> inputs) {
            assert(inputs.size() == input_layer.size(), "input vector's size must be the same size as the input layer's size");
            for (size_t i = 0; i < inputs.size(); i++) input_layer[i].set_value(inputs[i]);

            VectorXd input_values = initialize_vector(input_layer);
            VectorXd biases = initialize_vector(internal_layers[0], true);
            MatrixXd input_weights = initialize_weight_matrix(input_layer, internal_layers[0]);
            get_results(internal_layers[0], input_weights, input_values, biases);

            for (size_t layer = 1; layer < internal_layers.size(); layer++) {
                VectorXd value_vector = initialize_vector(internal_layers[layer-1]);
                VectorXd bias_vector = initialize_vector(internal_layers[layer], true);
                MatrixXd weight_matrix = initialize_weight_matrix(internal_layers[layer - 1], internal_layers[layer]);
                get_results(internal_layers[layer], weight_matrix, value_vector, bias_vector);
            }

            VectorXd final_input_values = initialize_vector(internal_layers[internal_layers.size() - 1]);
            VectorXd output_biases = initialize_vector(output_layer, true);
            MatrixXd final_input_weights = initialize_weight_matrix(internal_layers[internal_layers.size() - 1], output_layer);
            get_results(output_layer, final_input_weights, final_input_values, output_biases);
        }

        void train() {
            
        }

};