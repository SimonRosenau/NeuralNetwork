package de.rosenau.simon.neuralnetwork.impl;

import com.google.common.base.Preconditions;
import de.rosenau.simon.neuralnetwork.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Project created by Simon Rosenau.
 */

public class FeedForward extends NeuralNetwork {

    private Activation activation;
    private Neuron[][] neurons;

    private ReadWriteLock lock = new ReentrantReadWriteLock();

    public FeedForward(Activation activation, int... neurons) {
        Preconditions.checkNotNull(activation);
        Preconditions.checkNotNull(neurons);
        Preconditions.checkArgument(neurons.length >= 2, "You need at least an input and an output layer");

        this.activation = activation;
        this.neurons = new Neuron[neurons.length][];
        for (int i = 0; i < neurons.length; i++) {
            this.neurons[i] = new Neuron[neurons[i]];
            for (int o = 0; o < this.neurons[i].length; o++) {
                this.neurons[i][o] = i == 0 ? new Neuron() : new Neuron(neurons[i - 1]);
            }
        }
    }

    private static void addUp(double[] base, double[] addition) {
        Preconditions.checkArgument(base.length == addition.length, "Arrays not of the same lenght");
        for (int i = 0; i < base.length; i++) {
            base[i] += addition[i];
        }
    }

    private static void addUp(double[][] base, double[][] addition) {
        Preconditions.checkArgument(base.length == addition.length, "Arrays not of the same lenght");
        for (int i = 0; i < base.length; i++) {
            addUp(base[i], addition[i]);
        }
    }

    @Override
    public TrainingResult train(double[][] trainingInputs, double[][] trainingOutputs, TrainingProperties properties, TrainingObserver callback) {
        Preconditions.checkArgument(trainingInputs.length == trainingOutputs.length, "Invalid training sample sizes");
        Preconditions.checkNotNull(properties);
        Preconditions.checkArgument(properties.getLearningRate() > 0, "LearningRate must be greater than 0");
        Preconditions.checkArgument(properties.getMaxIterations() >= 0, "MaxIterations cannot be negative");
        Preconditions.checkArgument(properties.getMaxError() >= 0, "MaxError cannot be negative");
        Preconditions.checkArgument(properties.getBatches() > 0, "BatchSize must be greater than 0");
        Preconditions.checkArgument(trainingInputs.length % properties.getBatches() == 0, "BatchSize is not a divisor of training size");
        Preconditions.checkArgument(properties.getMaxError() != 0 || properties.getMaxIterations() != 0, "You have to specify MaxError or MaxIteration. Otherwise training will end in an infinite loop");

        lock.writeLock().lock();

        double[][][] inputBatches = new double[properties.getBatches()][][], outputBatches = new double[properties.getBatches()][][];

        for (int i = 0; i < properties.getBatches(); i++) {
            inputBatches[i] = Arrays.copyOfRange(trainingInputs, i * (trainingInputs.length / properties.getBatches()), (i + 1) * (trainingInputs.length / properties.getBatches()));
            outputBatches[i] = Arrays.copyOfRange(trainingOutputs, i * (trainingOutputs.length / properties.getBatches()), (i + 1) * (trainingOutputs.length / properties.getBatches()));
        }

        int iteration = 0;
        double error = properties.getMaxError();

        while((properties.getMaxIterations() == 0 || iteration < properties.getMaxIterations())
                && (properties.getMaxError() == 0 || properties.getMaxError() <= error)) {

            iteration++;

            // Iterate batches

            for (int bi = 0; bi < properties.getBatches(); bi++) {

                double[][] inputs = inputBatches[bi];
                double[][] outputs = outputBatches[bi];

                // Iterate samples

                Training[] trainings = new Training[this.neurons.length - 1];

                for (int i = 0; i < inputs.length; i++) {

                    double[] input = inputs[i];
                    double[] output = outputs[i];

                    double[] actual = compute(input);

                    // Calculate output layer
                    double[] activations = new double[output.length];

                    for (int o = 0; o < activations.length; o++) {
                        activations[o] = 2 * (actual[o] - output[o]);
                    }

                    // Backpropagation (without input layer)

                    for (int layer = this.neurons.length - 1; layer > 0; layer--) {
                        Training training = calculateLayer(layer, activations);

                        if (trainings[layer - 1] == null) {
                            trainings[layer - 1] = training;
                        } else {
                            addUp(trainings[layer - 1].weightDerivatives, training.weightDerivatives);
                            addUp(trainings[layer - 1].biasDerivatives, training.biasDerivatives);
                            addUp(trainings[layer - 1].prevActivationDerivatives, training.prevActivationDerivatives);
                        }

                        activations = training.prevActivationDerivatives;
                    }

                }

                // Adjust weights and biases

                double learningRate = properties.getLearningRate();

                for (int i = 1; i < neurons.length; i++) {
                    Neuron[] layer = neurons[i];
                    Training training = trainings[i - 1];

                    for (int o = 0; o < layer.length; o++) {
                        Neuron neuron = layer[o];

                        for (int p = 0; p < neuron.weights.length; p++) {
                            neuron.weights[p] -= training.weightDerivatives[o][p] * learningRate;
                        }

                        neuron.bias -= training.biasDerivatives[o] * learningRate;

                    }
                }
            }

            // Recalculate Error

            error = 0;

            for (int i = 0; i < trainingInputs.length; i++) {
                double[] inputs = trainingInputs[i];
                double[] outputs = trainingOutputs[i];

                double[] actual = compute(inputs);

                double current = 0;

                for (int o = 0; o < actual.length; o++) {
                    current += Math.abs(actual[o] - outputs[o]);
                }

                error += current / actual.length;
            }

            error /= trainingInputs.length;

            if (callback != null) callback.call(iteration, error);

        }

        lock.writeLock().unlock();

        return new TrainingResult(iteration, error);
    }

    private Training calculateLayer(int layer, double[] activationDerivatives) {
        Neuron[] neurons = this.neurons[layer];

        // dC0/dw(L)    = dz(L)/dw(L) * da(L)/dz(L) * dC0/da(L)
        //              = a(L-1) * sigm'(z(L)) * 2(a(L) - y)

        double[][] weightDerivatives = new double[neurons.length][];
        double[][] previousActivationDerivativesRaw = new double[neurons.length][];
        double[] biasDerivative = new double[neurons.length];

        // Iterate over neurons for weights and bias
        for (int i = 0; i < neurons.length; i++) {

            // Activation derivative
            double ca = activationDerivatives[i];
            // Activation over z
            double az = activation.derivative(neurons[i].z);

            // z over bias
            double zb = 1;

            weightDerivatives[i] = new double[neurons[i].weights.length];
            biasDerivative[i] = ca * az * zb;
            previousActivationDerivativesRaw[i] = new double[this.neurons[layer - 1].length];

            // Iterate over connections
            for (int o = 0; o < neurons[i].weights.length; o++) {

                // z over weight
                double zw = this.neurons[layer - 1][o].activation;

                // z over previousActivation
                double za = neurons[i].weights[o];

                // FillArray
                weightDerivatives[i][o] = ca * az * zw;
                previousActivationDerivativesRaw[i][o] = ca * az * za;
            }
        }

        // Add up previousActivation
        double[] previousActivationDerivative = new double[this.neurons[layer - 1].length];
        for (int i = 0; i < previousActivationDerivative.length; i++) {
            double derivative = 0;
            for (double[] aPreviousActivationDerivativesRaw : previousActivationDerivativesRaw) {
                derivative += aPreviousActivationDerivativesRaw[i];
            }
            previousActivationDerivative[i] = derivative;
        }

        return new Training(weightDerivatives, biasDerivative, previousActivationDerivative);
    }

    @Override
    public double[] compute(double[] input) {
        Preconditions.checkArgument(input.length == neurons[0].length, "Input array lenght does not match network input layer size");

        lock.readLock().lock();

        double[] output = new double[input.length];

        for (int i = 0; i < neurons.length; i++) {
            double[] newOutput = new double[neurons[i].length];
            for (int o = 0; o < neurons[i].length; o++) {
                if (i == 0) {
                    newOutput[o] = neurons[i][o].compute(input[o]);
                } else {
                    newOutput[o] = neurons[i][o].compute(output);
                }
            }
            output = newOutput;
        }

        lock.readLock().unlock();

        return output;
    }

    // Utils

    private static class Training {

        private double[][] weightDerivatives;
        private double[] biasDerivatives;
        private double[] prevActivationDerivatives;

        private Training(double[][] weightDerivatives, double[] biasDerivatives, double[] prevActivationDerivatives) {
            this.weightDerivatives = weightDerivatives;
            this.biasDerivatives = biasDerivatives;
            this.prevActivationDerivatives = prevActivationDerivatives;
        }

    }

    private class Neuron implements Serializable {

        private boolean input;

        private double[] weights;
        private double bias;

        private double z;
        private double activation;

        private Neuron() {
            this.input = true;
        }

        private Neuron(int size) {
            weights = new double[size];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = Math.random();
                //weights[i] = 0.5;
            }
            bias = 0;
            //bias = 1;
        }

        double compute(double input) {
            Preconditions.checkState(this.input, "Neuron is not an input neuron");
            this.activation = input;
            return input;
        }

        double compute(double[] activations) {
            Preconditions.checkState(!input, "Neuron is an input neuron");

            double a = 0;
            for (int i = 0; i < activations.length; i++) {
                a += weights[i] * activations[i];
            }
            a += bias;
            this.z = a;
            a = FeedForward.this.activation.activate(a);
            this.activation = a;
            return a;
        }

    }

}
