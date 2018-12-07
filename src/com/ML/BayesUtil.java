package com.ML;

import javafx.util.Pair;

import java.util.ArrayList;
import java.util.Collections;

@SuppressWarnings("Duplicates")
public class BayesUtil
{
    public static BayesParams bayesCrossValidate(String[] crosses, double[] smoothings)
    {
        double bestSmoothing = -1.0;
        FScore bestFScore = null;

        for (double smoothing : smoothings)
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

                //A new set is to be trained, so clear out the old.
                Example.resetAllKeys();
                //Read the examples from the chosen files
                ArrayList<Example> ex = GeneralUtil.readExamples(usedFiles);

                //Using the rate + loss for this run, epoch 10x over the targets
                Probabilities weights = bayesEpochs(1, ex, smoothing);

                FScore thisF = testFScore(weights, testFile);
                fSum.add(thisF);

            }

            fSum.divideBy(crosses.length);

            System.out.println("Smoothing " + smoothing + " has f-score: " + fSum.getfScore());

            if (bestFScore == null || fSum.getfScore() > bestFScore.getfScore())
            {
                bestFScore = new FScore(fSum);
                bestSmoothing = smoothing;
            }
        }

        System.out.println("Best smoothing " + bestSmoothing + " has values:"
                                   + "\nprecision: " + bestFScore.getPrecision()
                                   + "\nrecall: " + bestFScore.getRecall()
                                   + "\nfScore: " + bestFScore.getfScore());


        return new BayesParams(bestSmoothing);


    }


    /**
     * Runs Naive Bayes a specified number of times with a given learning rate and examples,
     * in order to learn a set of weights.
     * Weights are given random, tiny values to start.
     *
     * @return weights trained by running SVM a number of times
     */
    static Probabilities bayesEpochs(int epochs, ArrayList<Example> examples,
                                     double smoothing)
    {
        Probabilities weights = new Probabilities();

        return bayesEpochs(epochs, examples, weights, smoothing);

    }


    /**
     * Runs SVM a specified number of times with a given learning rate and examples,
     * in order to learn a set of weights.
     * The starting weights are as given.
     *
     * @return weights trained by running SVM a number of times
     */
    private static Probabilities bayesEpochs(int epochs, ArrayList<Example> examples, Probabilities weights,
                                             double smoothing)
    {

        Collections.shuffle(examples);

        //No need to reassign to weights,
        // since it is changed within the method.
        bayesEpoch(examples, weights, smoothing);

//            if (isTrain)
//                GeneralUtil.testVsDev(i + 1, weights);


        return weights;
    }


    /**
     * Runs Naive Bayes for one epoch. Updates weights.
     *
     * @param examples A list of Examples to run on.
     */
    private static void bayesEpoch(ArrayList<Example> examples, Probabilities weights,
                                   double smoothing)
    {
        //Fill in weights with probability values
        //Iterates over all possible feature-value pairs
        for (Pair<String, Double> eachPair : Example.getAllPossibilities())
        {
            String feat = eachPair.getKey();
            double value = eachPair.getValue();
            double chance = findLikelihood(examples, feat, value, true, smoothing);
            weights.put(feat, value, chance);
            double negChance = findLikelihood(examples, feat, value, false, smoothing);
            weights.put("-" + feat, value, negChance);
        }

        //Store the prior into the weights as well
        double prior = findPrior(examples);
        weights.setPrior(prior);

        return;
    }

    /**
     * Returns an FScore, given a set of weights
     *
     * @return an FScore object, holding precision, recall and the actual fScore
     */
    static FScore testFScore(Probabilities weights, String testFile)
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
     * Given an example and the probabilities of each feature-value pair, guess the example's sign.
     *
     * @return true if estimated chances of positive label >= .5
     */
    private static boolean sgn(Probabilities weights, Example eachEx)
    {
        double total = 1.0;
        double totalNeg = 1.0;

        for (String eachFeature : Example.getAllKeys())
        {
            double value;
            if (eachEx.hasKey(eachFeature))
                value = eachEx.get(eachFeature);
            else
                value = 0.0;

            double thisChance = weights.get(eachFeature, value);
            double thisNegChance = weights.get("-" + eachFeature, value);

            total *= thisChance;
            totalNeg *= thisNegChance;
        }

        double totalChance = total * weights.getPrior();
        double totalNegChance = totalNeg * (1 - weights.getPrior());
        boolean ret = (totalChance >= totalNegChance);
        return ret;
    }

    /**
     * For a given list of examples, returns numTrue / numTotal
     *
     * @return the probability that a randomly chosen example has a positive label
     */
    private static double findPrior(ArrayList<Example> examples)
    {
        double numTrue = 0.0;
        double numFalse = 0.0;

        for (Example each : examples)
        {
            if (each.getLabel())
                numTrue++;
            else
                numFalse++;
        }

        return numTrue / (numTrue + numFalse);
    }

    /**
     * For a given list of examples, returns P(feature = value | y = label)
     *
     * @return the probability that a randomly chosen example a certain value for a feature,
     * given that it has the given label
     */
    private static double findLikelihood(ArrayList<Example> examples, String feature, double value, boolean label, double smoothing)
    {
        double numTrue = 0.0;
        double numFalse = 0.0;
        double numLabelMatch = 0.0;

        for (Example each : examples)
        {
            if (each.getLabel() == label)
            {
                numLabelMatch++;

                if (each.get(feature) == value)
                    numTrue++;
                else
                    numFalse++;
            }
        }

        double likelihood = (numTrue + smoothing) / (numLabelMatch + (smoothing * examples.size()));
        return likelihood;
    }
}
