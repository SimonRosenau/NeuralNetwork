package de.rosenau.simon.neuralnetwork;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Project created by Simon Rosenau.
 */

@AllArgsConstructor
@Getter
public class TrainingResult {

    private int iterations;
    private double errorRemaining;

}
