package de.rosenau.simon.neuralnetwork;

import com.google.common.base.Preconditions;
import de.rosenau.simon.neuralnetwork.impl.FeedForward;

import java.io.IOException;

/**
 * Project created by Simon Rosenau.
 */

public class NetworkBuilder {

    private NetworkType type = NetworkType.FeedForward;
    private Activation activation = Activation.Sigmoid;
    private int[] neurons;

    public NetworkBuilder setType(NetworkType type) {
        Preconditions.checkNotNull(type);
        this.type = type;
        return this;
    }

    public NetworkBuilder setActivation(Activation function) {
        Preconditions.checkNotNull(function);
        this.activation = function;
        return this;
    }

    public NetworkBuilder setNeurons(int... neurons) {
        Preconditions.checkArgument(neurons.length > 1, "Network must at least have an input and an output layer");
        this.neurons = neurons;
        return this;
    }

    public NeuralNetwork build() {
        Preconditions.checkNotNull(neurons, "You must specify the neurons of the network");

        NeuralNetwork network = null;
        switch (type) {
            case FeedForward:
                network = new FeedForward(activation, neurons);
                break;
        }
        return network;
    }

    public NeuralNetwork fromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        return NeuralNetwork.fromBytes(bytes);
    }

}
