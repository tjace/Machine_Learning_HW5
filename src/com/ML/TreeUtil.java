package com.ML;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class TreeUtil
{
    public static ArrayList<Node> createTrees(String file, int depth) throws Exception
    {
        //Read all examples, and also fill in all possible feature-value pairs
        Example.resetAllKeys();
        ArrayList<Example> examples = GeneralUtil.readExamples(file);

        ArrayList<Node> nodes = new ArrayList<>(200);

        for (int i = 0; i < 200; i++)
        {
            //Create a separate list with 10% of the total examples
            ArrayList<Example> using = getRandoms(examples, 0.1);

            //This travel-size list of features is used by ID3
            HashSet<String> feats = new HashSet<>(Example.getAllKeys());

            //Construct the tree and return the root
            Node root = TreeUtil.ID3(using, feats, 1, depth);

            nodes.add(root);

        }

        return nodes;
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
     * Returns the tree's guess for the example.
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

            currentNode = currentNode.followPath(nextPath);
        }

        //debug.append(currentNode.name);
        //System.out.println(debug + "");
        return currentNode.getLabel();
    }
}
