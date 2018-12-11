package com.ML;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;

@SuppressWarnings("Duplicates")
class SimpleStochUtil
{
    @SuppressWarnings("Duplicates")
    static SVMParams SVMCrossValidate(String[] crosses, double[] SVMrates, double[] SVMlosses)
    {

        double bestRate = -1.0;
        double bestLoss = -1.0;
        FScore bestFScore = null;

        for (double loss : SVMlosses)
        {
            for (double rate : SVMrates)
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
                    Weight weights = stochEpochs(20, ex, rate, loss);

                    FScore thisF = testFScore(weights, testFile);
                    fSum.add(thisF);

                }

                fSum.divideBy(crosses.length);

                System.out.println("Starting rate " + rate + " + loss " + loss
                                           + " has f-score: " + fSum.getfScore());

                if (bestFScore == null || fSum.getfScore() > bestFScore.getfScore())
                {
                    bestFScore = new FScore(fSum);
                    bestRate = rate;
                    bestLoss = loss;
                }
            }
        }

        System.out.println("Best rate " + bestRate + " + loss " + bestLoss + " has values:"
                                   + "\nprecision: " + bestFScore.getPrecision()
                                   + "\nrecall: " + bestFScore.getRecall()
                                   + "\nfScore: " + bestFScore.getfScore());


        return new SVMParams(bestRate, bestLoss);
    }

    /**
     * Runs SVM a specified number of times with a given learning rate and examples,
     * in order to learn a set of weights.
     * Weights are given random, tiny values to start.
     *
     * @return weights trained by running SVM a number of times
     */
    static Weight stochEpochs(int epochs, ArrayList<Example> examples,
                              double startLearnRate, double loss)
    {
        Weight weights = new Weight();

        return stochEpochs(epochs, examples, weights, startLearnRate, loss);

    }

    /**
     * Runs SVM a specified number of times with a given learning rate and examples,
     * in order to learn a set of weights.
     * The starting weights are as given.
     *
     * @return weights trained by running SVM a number of times
     */
    private static Weight stochEpochs(int epochs, ArrayList<Example> examples, Weight weights,
                                      double startLearnRate, double loss)
    {
        //int strikes = 0;

        //while(strikes < 3)
        for (int i = 1; i < epochs + 1; i++)
        {
            Collections.shuffle(examples);

            //learnRate_t = learnRate_0 / (1 + t)
            double decayedRate = startLearnRate / (1 + i);

            //No need to reassign to weights,
            // since it is changed within the method.
            stochEpoch(examples, weights, decayedRate, loss);

//            if (isTrain)
//                GeneralUtil.testVsDev(i + 1, weights);
        }

        return weights;
    }

    /**
     * Runs SVM Stochastic Sub-Gradient. Updates weights.
     *
     * @param examples A list of Examples to run on.
     */
    private static void stochEpoch(ArrayList<Example> examples, Weight weights, double decayedRate, double loss)
    {
        for (Example ex : examples)
        {
            boolean sign = sgn(weights, ex);
            boolean actual = ex.getLabel();


            double y; //+1 if label is actually +, -1 if label is actually -
            if (ex.getLabel())
                y = 1.0;
            else
                y = -1.0;

            //Update each weight
            //if (y (w_T * x) <= 1):
            //   w <- w * (1 - decayedRate) + (y * decayedRate * loss * featureValue)
            //else
            //   w <- w * (1 - decayedRate)
            if (y * (sgnDouble(weights, ex)) <= 1)
            {

                for (String key : ex.getAllKeys())
                {
                    //weights.set(i, weights.get(i) + (y * learnRate * ex.get(i)));
                    double x;
                    if (ex.hasKey(key))
                        x = ex.get(key);
                    else
                        x = 0.0;

                    double newWeight = (weights.get(key) * (1 - decayedRate)) + (y * decayedRate * loss * x);
                    weights.put(key, newWeight);
                }

                //b <- b + (actualSign * learnRate)
                //weights.set(0, weights.get(0) + (y * learnRate));
//                double newB = weights.getB() + (y * decayedRate);
//                weights.setB(newB);
            }
            else
            {
                for (String key : Example.getAllKeys())
                {
                    double newWeight = (weights.get(key) * (1 - decayedRate));
                    weights.put(key, newWeight);
                }
            }
        }

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
     * Returns w_t * x:
     * b (stored in weights)
     * the weights w_t
     * an example x
     *
     * @param weights the weights to guess with
     * @param ex      the example to be guessed
     * @return (exampleFeatures * weights) + b
     */
    private static double sgnDouble(Weight weights, Example ex)
    {
        double sum = 0.0;

        for (String key : ex.getAllHeldKeys())
        {
            if (!(weights.hasKey(key)))
                weights.put(key, GeneralUtil.smallRandom());

            sum += (weights.get(key) * ex.get(key));
        }

//        sum += weights.getB(); //Don't forget to add b (for bonus lol)
        //Actually, ignore b for SVM

        return sum;
    }

    /**
     * Returns an estimated sign based on:
     * the weights
     * an example
     *
     * @param weights the weights to guess with
     * @param ex      the example to be guessed
     * @return true if (exampleFeatures * weights) + b >= 0, else false
     */
    private static boolean sgn(Weight weights, Example ex)
    {
        return (sgnDouble(weights, ex) >= 0);
    }

    static void printTestGuesses(Weight weights, ArrayList<Example> evalExamples, String evalIDFile, String outputFile) throws Exception {

        //This is what will write the output.
        PrintWriter writer = new PrintWriter(outputFile, StandardCharsets.UTF_8);
        writer.println("example_id,label");

        //This is for reading the IDs file
        BufferedReader evalReader = null;
        String IDLine;
        int lineNumber = 1;

        try {
            evalReader = new BufferedReader(new FileReader(evalIDFile));

            for (Example ex : evalExamples) {
                //Grab the example's ID (they are in order)
                IDLine = evalReader.readLine();
                String postLine;
                lineNumber++;

                boolean guess = sgn(weights, ex);
                if (guess)
                    postLine = IDLine + "," + "1";
                else
                    postLine = IDLine + "," + "0";
                System.out.println("Line " + lineNumber + ": " + postLine);
                writer.println(postLine);
            }

        } catch (
                FileNotFoundException e) {
            System.out.println("File " + evalIDFile + " not found.");
        } catch (
                IOException e) {
            e.printStackTrace();
        } finally {
            writer.close();

            if (evalReader != null) {
                try {
                    evalReader.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }


    }
}
