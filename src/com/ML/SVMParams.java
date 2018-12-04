package com.ML;

/**
 * This is a container class for two doubles, learnRate and lossTradeoff.
 */
public class SVMParams {

    private double learnRate;
    private double lossTradeoff;

    /**
     * Create a new SVMParams object.
     *
     * @param _learnRate    The learning rate
     * @param _lossTradeoff The tradeoff, C
     */
    public SVMParams(double _learnRate, double _lossTradeoff) {

        learnRate = _learnRate;
        lossTradeoff = _lossTradeoff;
    }

    /**
     * @return the learning rate
     */
    double getLearnRate() {
        return learnRate;
    }

    /**
     * @return the loss tradeoff, C
     */
    double getLossTradeoff() {
        return lossTradeoff;
    }


}
