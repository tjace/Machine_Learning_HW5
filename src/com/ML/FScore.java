package com.ML;

class FScore
{

    private double precision;
    private double recall;
    private double fScore;

    /**
     * Creates a new FScore object, given all of the values.
     *
     * @param _precision precision
     * @param _recall    recall
     * @param _fScore    given fScore
     */
    FScore(double _precision, double _recall, double _fScore)
    {
        precision = _precision;
        recall = _recall;
        fScore = _fScore;
    }

    /**
     * Create a new FScore object, calculating the fScore itself in-house
     *
     * @param _precision precision
     * @param _recall    recall
     */
    FScore(double _precision, double _recall)
    {
        precision = _precision;
        recall = _recall;

        if (precision == 0.0 || recall == 0.0)
            fScore = 0.0;
        else
            fScore = 2 * ((precision * recall) / (precision + recall));
    }

    /**
     * Create a new FScore object as a copy of other
     *
     * @param other the FScore object to copy
     */
    FScore(FScore other)
    {
        precision = other.getPrecision();
        recall = other.getRecall();
        fScore = other.getfScore();
    }


    /**
     * Create a new FScore object with all values set to 0
     */
    FScore()
    {
        precision = 0.0;
        recall = 0.0;
        fScore = 0.0;
    }

    double getPrecision()
    {
        return precision;
    }

    double getRecall()
    {
        return recall;
    }

    double getfScore()
    {
        return fScore;
    }

    /**
     * Add the values in another FScore to this one
     *
     * @param other the FScore with teh values to add into this one
     */
    void add(FScore other)
    {
        precision += other.getPrecision();
        recall += other.getRecall();
        fScore += other.getfScore();
    }

    /**
     * Divide all of the values in this by the divisor
     * Useful for cutting this down to average after adding in a few other FScore objects
     *
     * @param divisor the number to divide all values by
     */
    void divideBy(int divisor)
    {
        precision /= divisor;
        recall /= divisor;
        fScore /= divisor;
    }
}
