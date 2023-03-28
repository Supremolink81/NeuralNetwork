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
        float learn_rate;

        /// <summary>
        /// Activation function to make the input between 0 and 1.
        /// </summary>
        /// <param name="val">The activation of a neuron</param>
        inline double sigmoid(double val) {

            return 1.0 / (1.0 + std::exp(-val));

        }

        /// <summary>
        /// Determines the cost of a single test in the neural network.
        /// </summary>
        inline double cost() {

            double total_cost = 0;

            for (size_t i = 0; i < output_layer.size(); i++) {

                double cost = output_layer[i].get_value() - 1.0;
                total_cost += cost * cost;

            }

            return total_cost;
        }

        /// <summary>
        /// Initializes a matrix of neuron connection weights.
        /// </summary>
        /// <param name="input">: Current layer neurons</param>
        /// <param name="output">: Next layer neurons</param>
        /// <returns>The weight matrix</returns>
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

        /// <summary>
        /// Calculate the activation of the next layer using the weight matrix, 
        /// values of the current layer, and the bias.
        /// </summary>
        /// <param name="output">: Output layer reference</param>
        /// <param name="weights">: The weight matrix</param>
        /// <param name="values">: The value vector</param>
        /// <param name="biases">: The bias vector</param>
        void get_results(std::vector<neuron>& output, MatrixXd& weights, VectorXd& values, VectorXd& biases) {

            VectorXd results_vec = (weights * values) + biases;

            for (size_t i = 0; i < output.size(); i++) {

                output[i].set_value(sigmoid(results_vec[i]));

            }

        }

    public:

        /// <summary>
        /// Initializes a new neural network, given sizes for each layer.
        /// </summary>
        /// <param name="input_size">: size of input layer</param>
        /// <param name="layers">: number of internal layers in the neural network</param>
        /// <param name="layer_size">: size of each internal layer</param>
        /// <param name="output_size">: size of output layer</param>
        network(size_t input_size, size_t layers, size_t layer_size, size_t output_size, float learning_rate) {

            assert(input_size > 0 && layers > 0 && layer_size > 0 && output_size > 0, "all input sizes must be at least 1");

            //Initialize input layer
            input_layer.resize(input_size, neuron(layer_size));

            //Initialize each internal layer (excluding last layer)
            internal_layers.resize(layers);

            for (size_t i = 0; i < layers-1; i++) {
                internal_layers[i].resize(layer_size, neuron(layer_size));
            }

            internal_layers[layers - 1].resize(layer_size);

            //Last internal layer's weights depend on the side of the output layer
            for (neuron& n : internal_layers[layers - 1]) {

                n = neuron(output_size);

            }

            output_layer.resize(output_size);

            learn_rate = learning_rate;

        }

        /// <summary>
        /// Sets the values of the output layer, depending on the inputs given.
        /// 
        /// This function will initialize the values of each input neuron with the values
        /// in the input list, then set the values of all layers in the network based on the weights, 
        /// values, and biases of the neurons. 
        /// 
        /// At the end of this function, the output layer's neurons will all have their 
        /// activations set to various numbers between 0 and 1.
        /// 
        /// The amount of inputs must be equal to the input layer's size.
        /// </summary>
        /// <param name="inputs">: The values to place in the input layer</param>
        void forward(std::vector<double> inputs) {

            assert(inputs.size() == input_layer.size(), "input vector's size must be the same size as the input layer's size");

            for (size_t i = 0; i < inputs.size(); i++) {

                input_layer[i].set_value(inputs[i]);

            }
            
            // initialize first layer to input layer values
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


        /// <summary>
        /// Implementation of backpropagation using gradient descent.
        /// 
        /// This function will calculate the gradients at all layers of the network,
        /// then update the weights and biases accordingly. Used to train the network
        /// on a certain group of training data.
        /// </summary>
        /// <param name="expected_value">The index of the node that should be equal to 1</param>
        void backpropagate(int expected_value) {

            std::vector<std::vector<double>> internal_gradients(internal_layers.size(), std::vector<double>(internal_layers[0].size(), 0));
            std::vector<double> output_gradients(output_layer.size());
            std::vector<double> input_gradients(output_layer.size());

            //calculate output layer gradients

            for (size_t i = 0; i < output_gradients.size(); i++) {

                bool active = (expected_value == i);

                output_gradients[i] = (active - output_layer[i].get_value()) * output_layer[i].get_value() * (1 - output_layer[i].get_value());

            }

            //calculate internal layer weight gradients

            for (size_t i = 0; i < internal_layers[0].size(); i++) { //each neuron in the last internal layer

                for (size_t j = 0; j < output_layer.size(); j++) { //each neuron in the output layer

                    double gradient = output_gradients[j];
                    double connection_weight = internal_layers[internal_layers.size() - 1][i].get_weight(j);
                    internal_gradients[internal_layers.size() - 1][i] += gradient * connection_weight;

                }

                double activation = internal_layers[internal_layers.size() - 1][i].get_value();
                internal_gradients[internal_layers.size() - 1][i] *= activation * (1 - activation);

            }

            for (int l = internal_layers.size() - 2; l >= 0; l--) { //each internal layer in the network (except last)
                for (size_t i = 0; i < internal_layers[0].size(); i++) { //each neuron in the current internal layer
                    for (size_t j = 0; j < internal_layers[0].size(); j++) { //each neuron in the next internal layer
                        double gradient = internal_gradients[l + 1][j];
                        double connection_weight = internal_layers[l][i].get_weight(j);
                        internal_gradients[l][i] += gradient * connection_weight;
                    }
                    double activation = internal_layers[l][i].get_value();
                    internal_gradients[l][i] *= activation * (1 - activation);
                }
            }

            //calculate input gradients

            for (size_t i = 0; i < input_layer.size(); i++) { //each neuron in the input layer

                for (size_t j = 0; j < internal_layers[0].size(); j++) { //each neuron in the first internal layer

                    double gradient = internal_gradients[0][j];

                    double connection_weight = input_layer[i].get_weight(j);

                    input_gradients[i] += gradient * connection_weight;

                }

                double activation = input_layer[i].get_value();
                input_gradients[i] *= activation * (1 - activation);

            }

            //update internal layer weights and biases

            for (size_t i = 0; i < internal_layers[0].size(); i++) { //each neuron in the last internal layer

                double bias = output_layer[0].get_bias();
                double gradient = internal_gradients[internal_layers.size() - 1][i];

                for (size_t j = 0; j < output_layer.size(); j++) { //each neuron in the output layer

                    double weight = internal_layers[internal_layers.size() - 1][i].get_weight(j);
                    double activation_value = output_layer[j].get_value();

                    internal_layers[internal_layers.size() - 1][i].set_weight(j, weight - learn_rate * activation_value * gradient);

                    output_layer[j].set_bias(bias - learn_rate * gradient);

                }

            }

            for (int l = internal_layers.size() - 2; l >= 0; l--) { //each internal layer in the network (except last)

                for (size_t i = 0; i < internal_layers[0].size(); i++) { //each neuron in the current internal layer 

                    double gradient = internal_gradients[l][i];
                    double bias = internal_layers[l][i].get_bias();

                    for (size_t j = 0; j < internal_layers[0].size(); j++) { //each neuron in the next internal layer

                        double weight = internal_layers[l][i].get_weight(j);
                        double activation_value = internal_layers[l + 1][j].get_value();

                        internal_layers[l][i].set_weight(j, weight - learn_rate * activation_value * gradient);
                        internal_layers[l][i].set_bias(bias - learn_rate * gradient);

                    }

                }

            }

            //update input layer weights and biases

            for (size_t i = 0; i < input_layer.size(); i++) { //each neuron in the input layer

                double bias = input_layer[0].get_bias();
                double gradient = input_gradients[i];

                for (size_t j = 0; j < internal_layers[0].size(); j++) { //each neuron in the first internal layer

                    double weight = input_layer[i].get_weight(j);
                    double activation_value = internal_layers[0][j].get_value();

                    input_layer[i].set_weight(j, weight - learn_rate * activation_value * gradient);
                    internal_layers[0][j].set_bias(bias - learn_rate * gradient);

                }

            }

        }

        /// <summary>
        /// Trains the neural network on a group of input-output pairs.
        /// </summary>
        /// <param name="inputs">The inputs to the input layer of the neural network</param>
        /// <param name="expected_values">The expected output of the neural network</param>
        /// <param name="max_cost">The maximum cost the network will tolerate</param>
        /// <param name="max_epochs">The maximum amount of epochs to train the network</param>
        void train(std::vector<std::vector<double>> inputs, std::vector<int> expected_values, float max_cost, int max_epochs) {

            double c = 0.0;

            for (int e = 0; e < max_epochs; e++) {

                for (int i = 0; i < inputs.size(); i++) {

                    forward(inputs[i]);
                    backpropagate(expected_values[i]);
                    c = cost();

                }

                std::cout << "Epoch " << e + 1 << ", Cost: " << c << std::endl;
                if (c <= max_cost) break;

            }

        }

};