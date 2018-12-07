package com.ML;

import java.util.ArrayList;

public class Main
{

    //Hyper parameter choices for SVM
    private static final double[] SVMrates = new double[]{1.0, 0.1, 0.01, 0.001, 0.0001};
    private static final double[] SVMlosses = new double[]{10.0, 1.0, 0.1, 0.01, 0.001, 0.0001};

    //Hyper parameter choices for logistic regression
    private static final double[] LRrates = new double[]{1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001};
    private static final double[] LRtradeoffs = new double[]{0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0};

    //Hyper parameter choices for Naive Bayes
    private static final double[] bayesSmoothings = new double[]{0.0, 2.0, 1.5, 1.0, 0.5};

    //Hyper parameter choices for SVM over Trees
    private static final double[] treeDepths = new double[]{4, 8, 12, 16};
    private static final double[] treeRates = new double[]{1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001};
    private static final double[] treeLosses = new double[]{10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001};
    private static SVMParams bestSVMparams;

    //Test and training data
    private static final String trainFile = "src/data/train.liblinear";
    private static final String testFile = "src/data/test.liblinear";

    //The cross-validation data
    private static final String[] crosses = new String[]{
            "src/data/CVSplits/training00.data", "src/data/CVSplits/training01.data", "src/data/CVSplits/training02.data",
            "src/data/CVSplits/training03.data", "src/data/CVSplits/training04.data"};

    /**
     * Here we go, the main shebang.
     */
    public static void main(String[] args) throws Exception
    {
        //
        // simple stochastic sub-gradient descent version algorithm SVM
        //
        simpleStochastic();

        //
        //Logistic Regression classifier created with stochastic gradient descent
        //
        logisticRegression();

        //
        //Naive Bayes
        //
        naiveBayes();

        //
        //SVM over trees
        //
        SVMTrees();
    }


    /**
     * Run simple stochastic SVM with sub-gradients:
     * <p>
     * Find best parameters via cross validation,
     * then use them to learn weights on a training file and test on a test file.
     */
    private static void simpleStochastic()
    {
        //First, cross validate for the best hyper-params
        SVMParams params = SimpleStochUtil.SVMCrossValidate(crosses, SVMrates, SVMlosses);
        bestSVMparams = params;

        //With best params, train a new weightset on the .train file 20x
        //A new set is to be trained, so clear out the old.
        Example.resetAllKeys();
        ArrayList<Example> ex = GeneralUtil.readExamples(trainFile);
        Weight weights = SimpleStochUtil.stochEpochs(20, ex, params.getLearnRate(), params.getLossTradeoff());

        //then test for F1-score on the .test file.
        FScore score = SimpleStochUtil.testFScore(weights, testFile);

        //Finally, print the result.
        System.out.println("\n~~~~~~~~~~ RESULTS - SVM ~~~~~~~~~~");
        System.out.println("Best rate " + params.getLearnRate()
                                   + " + loss " + params.getLossTradeoff() + " has values vs test:"
                                   + "\nprecision: " + score.getPrecision()
                                   + "\nrecall: " + score.getRecall()
                                   + "\nfScore: " + score.getfScore() + "\n\n\n");


    }

    private static void logisticRegression()
    {
        //First, cross validate for the best hyper params
        LRParams params = LogisticRegressionUtil.LRCrossValidate(crosses, LRrates, LRtradeoffs);

        //With the best params, train a new weightset on the .train file 20x
        //A new set is to be trained, so clear out the old.
        Example.resetAllKeys();
        ArrayList<Example> ex = GeneralUtil.readExamples(trainFile);
        Weight weights = LogisticRegressionUtil.LREpochs(4, ex, params.getLearnRate(), params.getTradeoff());

        //Then test for F1-score on the .test file
        FScore score = LogisticRegressionUtil.testFScore(weights, testFile);

        //Finally, print the result.
        System.out.println("\n~~~~~~~~~~ RESULTS - Logistic Regression ~~~~~~~~~~");
        System.out.println("Best rate " + params.getLearnRate()
                                   + " + tradeoff " + params.getTradeoff() + " has values vs test:"
                                   + "\nprecision: " + score.getPrecision()
                                   + "\nrecall: " + score.getRecall()
                                   + "\nfScore: " + score.getfScore() + "\n\n\n");
    }

    private static void naiveBayes()
    {
        //First, cross validate for the best hyper params
        BayesParams params = BayesUtil.bayesCrossValidate(crosses, bayesSmoothings);

        //A new set is to be trained, so clear out the old.
        Example.resetAllKeys();
        ArrayList<Example> ex = GeneralUtil.readExamples(trainFile);

        //With the best params, train a set of probabilities on the train file
        Probabilities weights = BayesUtil.bayesEpochs(1, ex, params.getSmoothing());

        //Then test for F1-score on the .test file
        FScore score = BayesUtil.testFScore(weights, testFile);

//Finally, print the result.
        System.out.println("\n~~~~~~~~~~ RESULTS - BAYES ~~~~~~~~~~");
        System.out.println("Best smoothing " + params.getSmoothing()
                                   + " has values vs test:"
                                   + "\nprecision: " + score.getPrecision()
                                   + "\nrecall: " + score.getRecall()
                                   + "\nfScore: " + score.getfScore() + "\n\n\n");
    }

    /**
     * First, create a list of 200 trees of small depth
     * Then, run each example through the trees and get their outputs
     * Finally, run SVM
     */
    private static void SVMTrees() throws Exception
    {

        //First, create the list of trees from the training file
        ArrayList<Node> trees = TreeUtil.createTrees(trainFile, 4);


        //Create find the best params using the output of the trees
        //SVMParams params = TreeUtil.crossValidate(trees, trainFile, treeRates, treeLosses);

        //Create a set of examples from the outputs of the trees
        Example.resetAllKeys();
        ArrayList<Example> ex = TreeUtil.createExamples(trees, trainFile);

        //With the best params, train a weightset on the training trees
        Weight weights = SimpleStochUtil.stochEpochs(20, ex, bestSVMparams.getLearnRate(), bestSVMparams.getLossTradeoff());

        //then test for F1-score on the .test file.
        FScore score = SimpleStochUtil.testFScore(weights, testFile);

        //Finally, print the result.
        System.out.println("\n~~~~~~~~~~ RESULTS - SVM ~~~~~~~~~~");
        System.out.println("Best depth " + 4 + " rate " + bestSVMparams.getLearnRate()
                                   + " + loss " + bestSVMparams.getLossTradeoff() + " has values vs test:"
                                   + "\nprecision: " + score.getPrecision()
                                   + "\nrecall: " + score.getRecall()
                                   + "\nfScore: " + score.getfScore() + "\n\n\n");
    }
}
