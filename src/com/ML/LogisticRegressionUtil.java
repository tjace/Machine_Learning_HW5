package com.ML;

import java.util.ArrayList;
import java.util.Collections;

@SuppressWarnings("ALL")
public class LogisticRegressionUtil
{
    @SuppressWarnings("Duplicates")
    static LRParams LRCrossValidate(String[] crosses, double[] LRrates, double[] LRtradeoffs)
    {
        double bestRate = -1.0;
        double bestTradeoff = -1.0;
        FScore bestFScore = null;

        for (double tradeoff : LRtradeoffs)
        {
            for (double rate : LRrates)
            {

                FScore fSum = new FScore();

                //Do a run for each cross-testing file to be used and sum up the f-score
                for (int i = 0; i < crosses.length; i++)
                {

                    //Use all of the files but one
                    String[] usedFiles = new String[crosses.length - 1];
                    String testFile = "";
                    int found = 0;
                    for (int j = 0; j < crosses.length; j++)
                    {
                        if (j == i)
                        {
                            found = 1;
                            testFile = crosses[j];
                            continue;
                        }
                        usedFiles[j - found] = crosses[j];
                    }

                    //Read the examples from the chosen files
                    ArrayList<Example> ex = GeneralUtil.readExamples(usedFiles);

                    //Using the rate + loss for this run, epoch 10x over the targets
                    Weight weights = LREpochs(20, ex, rate, tradeoff);

                    FScore thisF = testFScore(weights, testFile);
                    fSum.add(thisF);

                }

                fSum.divideBy(crosses.length);

                System.out.println("Starting rate " + rate + " + tradeoff " + tradeoff
                                           + " has f-score: " + fSum.getfScore());

                if (bestFScore == null || fSum.getfScore() > bestFScore.getfScore())
                {
                    bestFScore = new FScore(fSum);
                    bestRate = rate;
                    bestTradeoff = tradeoff;
                }
            }
        }

        System.out.println("Best rate " + bestRate + " + tradeoff " + bestTradeoff + " has values:"
                                   + "\nprecision: " + bestFScore.getPrecision()
                                   + "\nrecall: " + bestFScore.getRecall()
                                   + "\nfScore: " + bestFScore.getfScore());


        return new LRParams(bestRate, bestTradeoff);
    }

    /**
     * Runs Stochastic Gradient Logistic Regression a specified number of times with a given learning rate and examples,
     * in order to learn a set of weights.
     * Weights are given random, tiny values to start.
     *
     * @return weights trained by running SG-LR a number of times
     */
    static Weight LREpochs(int epochs, ArrayList<Example> examples,
                           double startLearnRate, double tradeoff)
    {
        Weight weights = new Weight();

        return LREpochs(epochs, examples, weights, startLearnRate, tradeoff);
    }

    /**
     * Runs Stochastic Gradient Logistic Regression a specified number of times with a given learning rate and examples,
     * in order to learn a set of weights.
     * The starting weights are given.
     *
     * @return weights trained by running SG-LR a number of times
     */
    private static Weight LREpochs(int epochs, ArrayList<Example> examples, Weight weights,
                                   double startLearnRate, double tradeoff)
    {
        //int strikes = 0;

        //while(strikes < 3)
        for (int i = 0; i < epochs; i++)
        {
            Collections.shuffle(examples);

            //learnRate_t = learnRate_0 / (1 + t)
            double decayedRate = startLearnRate / (1 + i);

            //No need to reassign to weights,
            // since it is changed within the method.
            LREpoch(examples, weights, decayedRate, tradeoff);

//            if (isTrain)
//                GeneralUtil.testVsDev(i + 1, weights);
        }

        return weights;
    }

    /**
     * Runs Stochastic Gradient Logistic Regression.
     * Updates weights.
     */
    private static void LREpoch(ArrayList<Example> examples, Weight weights, double decayedRate, double tradeoff)
    {

        for (Example ex : examples)
        {
            boolean actual = ex.getLabel();

            double y; //+1 if label is actually +, -1 if label is actually -
            if (ex.getLabel())
                y = 1.0;
            else
                y = -1.0;

            //Update each weight
            //if (y (w_T * x) <= 1):
            //   w  <--  w - (1/ln(1 + exp(-y * w * x)) + (exp( -y * w * x)) + (-yixi) + (wt-1)2/σ^2 )
            for (String key : Example.getAllKeys())
            {
                double w = weights.get(key);
                double x;
                if (ex.hasKey(key))
                    x = ex.get(key);
                else
                    x = 0.0;

                //double newWeight = w - decayedRate * ( (1 / (Math.log(1 + Math.exp(-y * w * x))))
                 //       + Math.exp(-y * w * x) + (-y * x) + (2 * w / tradeoff * tradeoff));

                double newWeight = w + decayedRate * y * tradeoff * x;

                weights.put(key, newWeight);
            }
        }

        return;
    }


    static FScore testFScore(Weight weights, String testFile)
    {
        double precision;
        double recall;
        double fScore;

        double truePos = 0.0;
        double falsePos = 0.0;
        double falseNeg = 0.0;


        ArrayList<Example> testExamples = GeneralUtil.readExamples(testFile);

        for (Example eachEx : testExamples)
        {

            boolean guess = sgn(weights, eachEx);
            boolean actual = eachEx.getLabel();

            if (guess != actual)
            {
                if (guess)
                    falsePos++;
                else
                    falseNeg++;
            }
            else
            {
                if (guess) //Guess == actual ==true
                    truePos++;
            }
        }

        if ((truePos + falsePos) != 0)
            precision = truePos / (truePos + falsePos);
        else
            precision = 0;

        if ((truePos + falseNeg) != 0)
            recall = truePos / (truePos + falseNeg);
        else
            recall = 0;

        FScore score_obj = new FScore(precision, recall);

        return score_obj;
    }

    /**
     * Returns an estimated sign based on:
     * the weights
     * an example
     *
     * @param weights the weights to guess with
     * @param ex      the example to be guessed
     * @return true if (exampleFeatures * weights) >= 0, else false
     */
    private static boolean sgn(Weight weights, Example ex)
    {
        double sum = 0.0;

        for (String key : ex.getAllHeldKeys())
        {
            if (!(weights.hasKey(key)))
                weights.put(key, GeneralUtil.smallRandom());

            sum += (weights.get(key) * ex.get(key));
        }

        return (sum >= 0);
    }

}
