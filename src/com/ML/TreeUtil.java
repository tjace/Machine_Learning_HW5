package com.ML;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class TreeUtil
{
    /**
     * Creates 200 trees, each set constructed from 10% of the original file
     *
     * @param file
     * @param depth
     * @return
     * @throws Exception
     */
    public static ArrayList<Node> createTrees(String file, int depth, boolean presence) throws Exception
    {
        //Read all examples, and also fill in all possible feature-value pairs
        Example.resetAllKeys();
        ArrayList<Example> examples = GeneralUtil.readExamples(file, presence);

        ArrayList<Node> nodes = new ArrayList<>(200);
        HashSet<String> usedFeats = new HashSet<>(200);


        for (int i = 0; i < 200; i++)
        {
            //Create a separate list with 10% of the total examples
            ArrayList<Example> using = getRandoms(examples, 0.1);

            //This travel-size list of features is used by ID3
            HashSet<String> feats = new HashSet<>(Example.getSomeKeys());
            //Don't look at any features that have already been featured as pivots
            feats.removeAll(usedFeats);

            //Construct the tree and return the root
            Node root = TreeUtil.ID3(using, feats, 1, depth);

            nodes.add(root);
            usedFeats.addAll(root.getAllNames());

        }

        return nodes;
    }

    public static ArrayList<Node> createTrees(String file, int depth) throws Exception
    {
        return createTrees(file, depth, false);
    }

    /**
     * @param examples The list of examples to take from
     * @param v        the percent of the list to take
     * @return v of examples, randomly
     */
    private static ArrayList<Example> getRandoms(ArrayList<Example> examples, double v)
    {
        Random rng = new Random();
        int count = (int) (Math.floor(examples.size() * v));

        ArrayList<Example> ret = new ArrayList<>(count);
        HashSet<Integer> alreadyDone = new HashSet<>();

        for (int i = 0; i < count; i++)
        {
            int index = rng.nextInt(examples.size());
            Example item = examples.get(index);

            ret.add(item);
            alreadyDone.add(index);
        }

        return ret;
    }


    /**
     * if maxDepth == -1, there is no max.
     */
    static Node ID3(ArrayList<Example> examples, HashSet<String> features, int depth, int maxDepth) throws Exception
    {
        //If all shrooms have the same label, return a leaf with that label
        int sameyLabel = checkAllSameLabel(examples);
        if (sameyLabel == 1)
            return new Node("+1", depth, true);
        if (sameyLabel == -1)
            return new Node("-1", depth, true);
        if (sameyLabel != 0)
            throw new Exception("Samey label not 0, 1 or -1: " + sameyLabel);


        //If this is the lowest a node can be, it'll have to be a leaf.
        if (maxDepth != -1 && depth == maxDepth)
        {
            String commonLabel = findCommonLabel(examples);
            return new Node(commonLabel, depth, true);
        }


        //If this is as far as the features go (al have been checked already), make leaf node using the most common label.
        if (features.isEmpty())
        {
            String commonLabel = findCommonLabel(examples);
            return new Node(commonLabel, depth, true);
        }

        //Determine best feature to discriminate by at this point
        //Using InfoGain
        String bestFeature = "warui desu.";

        double maxGain = 0;
        for (String eachFeat : features)
        {
            double infoGain = infoGain(eachFeat, examples);

            if (infoGain > maxGain)
            {
                maxGain = infoGain;
                bestFeature = eachFeat;
            }
        }


        //Remove the used feature from later branches
        HashSet<String> nextFeatures = new HashSet<>(features);
        nextFeatures.remove(bestFeature);

        Node thisNode = new Node(bestFeature, depth, false);

        //For each value of the feature (e.g. sweet, spicy, mild)
        //enact again ID3 using only the surviving shrooms

        for (double nextAtt : Example.getAllValuesOfFeature(bestFeature))
        {
            //Construct Sv (subset of shrooms that have the specified value/attribute of the bestFeature)
            ArrayList<Example> nextShrooms = new ArrayList<>();

            for (Example shroom : examples)
            {
                if (shroom.hasKey(bestFeature))
                {
                    if (shroom.get(bestFeature) == nextAtt)
                    {
                        nextShrooms.add(shroom);
                    }
                }
            }

            Node nextNode;
            if (nextShrooms.size() == 0)
            {
                String commonLabel = findCommonLabel(examples);
                nextNode = new Node(commonLabel, depth + 1, true);
            }
            else
            {
                nextNode = ID3(nextShrooms, nextFeatures, depth + 1, maxDepth);
            }

            thisNode.add((int) nextAtt, nextNode);

        }

        return thisNode;
    }

    /**
     * Returns log base 2 of the input.
     */
    private static double logBase2(double input)
    {
        return Math.log(input) / Math.log(2);
    }

    /**
     * Returns the most common label.
     *
     * @param shrooms the set of examples to check for common label
     * @return either "+1" or "-1"
     */
    @SuppressWarnings("Duplicates")
    private static String findCommonLabel(ArrayList<Example> shrooms)
    {
        int pos = 0;
        int neg = 0;

        for (Example mush : shrooms)
        {
            boolean label = mush.getLabel();

            if (label)
                pos++;
            else
                neg++;
        }

        if (neg > pos)
            return "-1";
        else
            return "+1";
    }

    /**
     * Check if all examples remaining are the same label (+ or -)
     *
     * @param shrooms The list of examples to compare labels with
     * @return 1 if all true, -1 if all false, 0 if not all same.
     */
    private static int checkAllSameLabel(ArrayList<Example> shrooms)
    {

        //find the label of the first example in the list
        boolean label = shrooms.get(0).getLabel();

        //If any following don't match the first example, return 0
        for (Example shroom : shrooms)
        {
            if (!(shroom.getLabel() == label))
                return 0;
        }

        //If all are the same, return +1/-1 depending on if all are true/false
        if (label)
            return 1;
        else
            return -1;
    }


    /**
     * Calculate the information gain of a feature across the given shrooms
     * Gain(for set S, attribute A)  =  Entropy(on set S) - SUM_for_all_v_in_A( (|Sv| / |S|) * Entropy(on set Sv) )
     * Sv = subset of examples where attribute A has value V
     *
     * @param feature the feature to check for information gain
     * @param shrooms the set of current shrooms
     * @return the amount of information gain
     */
    private static double infoGain(String feature, ArrayList<Example> shrooms)
    {
        double bigEntropy = entropy(shrooms); //Entropy of set S
        double expectedEntropy = 0; //Entropy(on set Sv)

        for (double att : Example.getAllValuesOfFeature(feature))
        {

            //get the subset of shrooms with each value of the feature
            ArrayList<Example> nextShrooms = new ArrayList<>();
            for (Example shroom : shrooms)
            {
                if (shroom.hasKey(feature))
                {
                    if (shroom.get(feature) == att)
                    {
                        nextShrooms.add(shroom);
                    }
                }
            }

            double thisEntropy = entropy(nextShrooms);

            expectedEntropy += (((double) (nextShrooms.size()) / (double) (shrooms.size())) * thisEntropy);


        }

        return bigEntropy - expectedEntropy;
    }

    /**
     * Entropy = -(pPos)log(pPos) - (pNeg)log(pNeg)
     *
     * @param shrooms Set to measure entropy across
     * @return a double representing entropy
     */
    @SuppressWarnings("Duplicates")
    private static double entropy(ArrayList<Example> shrooms)
    {
        int pPos = 0;
        int pNeg = 0;

        double total = shrooms.size();

        for (Example shroom : shrooms)
        {
            boolean value = shroom.getLabel();
            if (value)
                pPos++;
            else
                pNeg++;
        }

        double entropy = 0;

        double proportion;
        double logResult;
        //-(pPos)log(pPos)
        if (pPos > 0)
        {
            proportion = (pPos / total);
            logResult = logBase2(proportion);
            entropy -= (proportion * logResult);
        }

        if (pNeg > 0)
        {
            //- (pNeg)log(pNeg)
            proportion = (pNeg / total);
            logResult = logBase2(proportion);
            entropy -= (proportion * logResult);
        }

        return entropy;
    }

    /**
     * Given a file and set of trees, output a set of Examples that correspond to the output (1, 0) of the trees per example
     *
     * @return
     */
    public static ArrayList<Example> createExamples(ArrayList<Node> trees, String trainFile) throws Exception
    {
        ArrayList<Example> examples = GeneralUtil.readExamples(trainFile);

        ArrayList<Example> ret = new ArrayList<>();

        for (Example eachEx : examples)
        {
            Example ex = new Example();

            int count = 0;

            for (Node root : trees)
            {
                double res;

                boolean guess = sgn(root, ex);
                if (guess)
                    res = 1.0;
                else
                    res = 0.0;

                ex.add(Integer.toString(count), res);

                count++;
            }

            ret.add(ex);
        }

        return ret;
    }


    /**
     * Returns one tree's guess for the example.
     */
    static boolean sgn(Node root, Example ex) throws Exception
    {
        // StringBuilder debug = new StringBuilder();
        Node currentNode = root;

        while (!currentNode.isLeaf())
        {
            int nextPath;
            if (ex.hasKey(currentNode.name))
                nextPath = (ex.get(currentNode.name)).intValue();
            else
                nextPath = 0;

            //debug.append(currentNode.name).append(": ").append(nextPath).append(" ==> ");
            if(currentNode.followPath(nextPath) == null)
                System.out.println("why");

            currentNode = currentNode.followPath(nextPath);

        }

        //debug.append(currentNode.name);
        //System.out.println(debug + "");
        return currentNode.getLabel();
    }

    /**
     * Returns the average of many trees' guesses for the example.
     */
    static boolean sgn(ArrayList<Node> roots, Example ex) throws Exception
    {
        int pos = 0;
        int neg = 0;

        for (Node each : roots)
        {
            if (sgn(each, ex))
                pos++;
            else neg++;
        }

        if (pos >= neg)
            return true;
        else
            return false;
    }

    /**
     * Returns an FScore, given a set of weights
     *
     * @return an FScore object, holding precision, recall and the actual fScore
     */
    static FScore testFScore(ArrayList<Node> weights, String testFile) throws Exception
    {
        double precision;
        double recall;
        double fScore;

        double truePos = 0.0;
        double falsePos = 0.0;
        double falseNeg = 0.0;

        ArrayList<Example> testExamples = GeneralUtil.readExamples(testFile, true);

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

    static void printTestGuesses(ArrayList<Node> trees, ArrayList<Example> evalExamples, String evalIDFile, String outputFile) throws Exception {

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

                boolean guess = sgn(trees, ex);
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
