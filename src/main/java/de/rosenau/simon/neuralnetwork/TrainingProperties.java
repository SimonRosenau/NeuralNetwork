package de.rosenau.simon.neuralnetwork;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * Project created by Simon Rosenau.
 */

@Builder
@Getter
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class TrainingProperties {

    public static final TrainingProperties DEFAULT_PROPERTIES = new TrainingProperties(0.5, 0, 0.01, 1);

    @Builder.Default
    private double learningRate = 0.5;
    @Builder.Default
    private int maxIterations = 0;
    @Builder.Default
    private double maxError = 0;
    @Builder.Default
    private int batches = 1;

}
