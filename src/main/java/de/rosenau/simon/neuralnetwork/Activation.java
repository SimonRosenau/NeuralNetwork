package de.rosenau.simon.neuralnetwork;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Project created by Simon Rosenau.
 */

@AllArgsConstructor
@Getter(AccessLevel.PACKAGE)
public enum Activation {

    ArcTan(Math::atan, a -> 1 / (Math.pow(a, 2) + 1)),
    Gaussian(a -> Math.exp(-Math.pow(a, 2)), a -> -2 * a * Math.exp(-Math.pow(a, 2))),
    Identity(a -> a, a -> 1),
    LeakyReLU(a -> a < 0 ? 0.01 * a : a, a -> a > 0 ? 1 : 0.01),
    ReLU(a -> a < 0 ? 0 : a, a -> a > 0 ? 1 : 0),
    Sigmoid(a -> 1 / (1 + Math.exp(-a)), a -> 1 / (1 + Math.exp(-a)) * (1 - 1 / (1 + Math.exp(-a)))),
    Sinc(a -> a == 0 ? 1 : Math.sin(a) / a, a -> a == 0 ? 0 : Math.cos(a) / a - Math.sin(a) / Math.pow(a, 2)),
    Sinusoid(Math::sin, Math::cos),
    SoftPlus(a -> Math.log(1 + Math.exp(a)), a -> 1 / (1 + Math.exp(-a))),
    SoftSign(a -> a / (1 + Math.abs(a)), a -> 1 / Math.pow(1 + Math.abs(a), 2)),
    Swish(a -> a * (1 / (1 + Math.exp(-a))), a -> a * (1 / (1 + Math.exp(-a))) + 1 / (1 + Math.exp(-a)) * (1 - a * (1 / (1 + Math.exp(-a))))),
    TanH(a -> (Math.exp(a) - Math.exp(-a)) / (Math.exp(a) + Math.exp(-a)), a -> 1 - Math.pow((Math.exp(a) - Math.exp(-a)) / (Math.exp(a) + Math.exp(-a)), 2));

    private Function activation;
    private Function derivative;

    public double activate(double a) {
        return activation.calculate(a);
    }

    public double derivative(double a) {
        return derivative.calculate(a);
    }

    private interface Function {
        double calculate(double a);
    }

}