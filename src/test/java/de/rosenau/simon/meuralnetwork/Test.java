package de.rosenau.simon.meuralnetwork;

import de.rosenau.simon.neuralnetwork.*;

import java.util.Arrays;

/**
 * Project created by Simon Rosenau.
 */

public class Test {

    public static void main(String[] args) {
        NeuralNetwork network = new NetworkBuilder().setType(NetworkType.FeedForward).setActivation(Activation.Sigmoid).setNeurons(2, 2, 1).build();

        double[][] inputs = new double[][]{
                new double[]{0, 0},
                new double[]{0, 1},
                new double[]{1, 0},
                new double[]{1, 1}
        };
        double[][] outputs = new double[][]{
                new double[]{0},
                new double[]{1},
                new double[]{1},
                new double[]{0}
        };

        for (int i = 0; i < inputs.length; i++) {
            System.out.println(Arrays.toString(network.compute(inputs[i])));
        }

        TrainingResult result = network.train(inputs, outputs, TrainingProperties.builder().maxError(0.01).build());
        System.out.println(result.getIterations() + " " + result.getErrorRemaining());

        System.out.println();
        for (int i = 0; i < inputs.length; i++) {
            System.out.println(Arrays.toString(network.compute(inputs[i])));
        }

    }

    public static void main1(String[] args) throws Exception {
        NeuralNetwork network = new NetworkBuilder().setType(NetworkType.FeedForward).setActivation(Activation.Sigmoid).setNeurons(784, 16, 16, 10).build();

        MNIST training = new MNIST("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
        MNIST test = new MNIST("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

        double[][] inputs = training.getData();
        double[][] outputs = convertLabelsToOutput(training.getLabels());

        test(network, test);

        network.train(inputs, outputs, TrainingProperties.builder().batches(60000).maxError(0.01).maxIterations(50).learningRate(0.1).build(), (iteration, error) -> System.out.println(iteration + " " + error));

        test(network, test);
    }

    public static void test(NeuralNetwork network, MNIST test) {
        double[][] inputs = test.getData();
        byte[] labels = test.getLabels();

        double right = 0;

        for (int i = 0; i < inputs.length; i++) {
            double[] output = network.compute(inputs[i]);
            int result = 0;
            for (int o = 0; o < output.length; o++) {
                if (output[o] > output[result]) result = o;
            }
            if (result == labels[i]) {
                right++;
            }
        }
        System.out.println("Right guesses: " + (int) right + " (" + (right / labels.length * 100) + "%)");
    }

    public static double[][] convertLabelsToOutput(byte[] labels) {
        double[][] doubles = new double[labels.length][10];
        for (int i = 0; i < labels.length; i++) {
            byte label = labels[i];
            doubles[i][label] = 1;
        }
        return doubles;
    }

}
