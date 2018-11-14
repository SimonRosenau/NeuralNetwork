package de.rosenau.simon.neuralnetwork;

import java.io.*;

/**
 * Project created by Simon Rosenau.
 */

public abstract class NeuralNetwork implements Serializable {

    public static NeuralNetwork fromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
        return (NeuralNetwork) new ObjectInputStream(new ByteArrayInputStream(bytes)).readObject();
    }

    public TrainingResult train(double[][] inputs, double[][] outputs) {
        return train(inputs, outputs, TrainingProperties.DEFAULT_PROPERTIES, null);
    }

    public TrainingResult train(double[][] inputs, double[][] outputs, TrainingProperties properties) {
        return train(inputs, outputs, properties, null);
    }

    public TrainingResult train(double[][] inputs, double[][] outputs, TrainingObserver callback) {
        return train(inputs, outputs, TrainingProperties.DEFAULT_PROPERTIES, callback);
    }

    public abstract TrainingResult train(double[][] inputs, double[][] outputs, TrainingProperties properties, TrainingObserver callback);

    public abstract double[] compute(double[] input);

    public byte[] toBytes() throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        new ObjectOutputStream(outputStream).writeObject(this);
        return outputStream.toByteArray();
    }

}
