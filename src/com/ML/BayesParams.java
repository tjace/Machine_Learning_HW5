package com.ML;

/**
 * This is a container class for a double, smoothing.
 */
public class BayesParams {

    private double smoothing;

    /**
     * Create a new SVMParams object.
     *
     * @param _smoothing    The smoothing
     */
    public BayesParams(double _smoothing) {
        smoothing = _smoothing;
    }

    /**
     * @return the smoothing
     */
    double getSmoothing() {
        return smoothing;
    }
}
