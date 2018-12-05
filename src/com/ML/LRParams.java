package com.ML;

/**
 * This is a container class for two doubles, learnRate and tradeoff.
 */
public class LRParams {

    private double learnRate;
    private double tradeoff;

    /**
     * Create a new SVMParams object.
     *
     * @param _learnRate    The learning rate
     * @param _Tradeoff The tradeoff, σ^2
     */
    public LRParams(double _learnRate, double _Tradeoff) {

        learnRate = _learnRate;
        tradeoff = _Tradeoff;
    }

    /**
     * @return the learning rate
     */
    double getLearnRate() {
        return learnRate;
    }

    /**
     * @return the tradeoff, σ^2
     */
    double getTradeoff() {
        return tradeoff;
    }


}
