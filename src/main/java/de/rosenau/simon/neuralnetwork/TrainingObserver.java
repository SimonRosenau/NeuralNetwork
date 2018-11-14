package de.rosenau.simon.neuralnetwork;

/**
 * Project created by Simon Rosenau.
 */

public interface TrainingObserver {

    void call(int iteration, double error);

}
