package com.ML;

import javafx.util.Pair;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * an Example's features array contains:
 * features[0]:         -1.0 or 1.0, corresponding with the label
 * features [1 - 19]:   a double, corresponding to the fields 1 - 19.  1 is the lowest field, and 19 is the highest.
 */
class Example
{

    private boolean label;
    private HashMap<String, Double> features;
    private static HashMap<String, Set<Double>> allKeys;

    /**
     * Create a new example based on this read-in line.
     */
    Example(String fullLine)
    {
        this(fullLine, false);
    }

    /**
     * Rather than storing a value for each feature, stores 1.0 for all values present
     * (assuming presence is true)
     */
    Example(String fullLine, boolean presence)
    {

        features = new HashMap<>();
        String[] pieces = fullLine.split(" ");

        for (String each : pieces)
        {
            switch (each)
            {
                case "-1":
                case "0":
                    label = false;
                    break;
                case "+1":
                case "1":
                    label = true;
                    break;
                default:
                    String[] splits = each.split(":"); //key:value

                    if (!presence)
                    {
                        features.put(splits[0], Double.parseDouble(splits[1]));
                        //Make sure allKeys has this feature-value, and also feature-0.0
                        addToAllKeys(splits[0], Double.parseDouble(splits[1]));
                        addToAllKeys(splits[0], 0.0);
                    }
                    else //If presence mode on, map all present features to 1.0.
                    {
                        features.put(splits[0], 1.0);
                        //Make sure allKeys has this feature-1.0, and also feature-0.0
                        addToAllKeys(splits[0], 1.0);
                        addToAllKeys(splits[0], 0.0);
                    }
            }
        }


    }

    /**
     * Creates a new empty example, to be built piece by piece.
     */
    Example()
    {
        features = new HashMap<>();
    }

    /**
     * Adds a new feature-value pair to this Example, and all Examples' static memory
     *
     * @param _key   feature
     * @param _value value
     */
    void add(String _key, double _value)
    {
        features.put(_key, _value);
        addToAllKeys(_key, _value);
    }


    /**
     * Ensure that allKeys has the feature and value stored
     *
     * @param feature
     * @param value
     */
    private void addToAllKeys(String feature, double value)
    {
        if (!allKeys.containsKey(feature))
            allKeys.put(feature, new HashSet<>());
        allKeys.get(feature).add(value);
        return;
    }


    Double get(String n)
    {
        return features.getOrDefault(n, 0.0);
    }

    boolean getLabel()
    {
        return label;
    }

    boolean hasKey(String key)
    {
        return features.containsKey(key);
    }

    Set<String> getAllHeldKeys()
    {
        return features.keySet();
    }

    /**
     * Deletes all stored keys from the static set of keys
     */
    static void resetAllKeys()
    {
        if (allKeys != null)
            allKeys.clear();
        else
            allKeys = new HashMap<>();
    }

    static Set<String> getAllKeys()
    {
        return allKeys.keySet();
    }

    static Set<Pair<String, Double>> getAllPossibilities()
    {
        HashSet<Pair<String, Double>> ret = new HashSet<>();

        for (String feat : allKeys.keySet())
        {
            for (double value : allKeys.get(feat))
            {
                ret.add(new Pair<>(feat, value));
            }
        }

        return ret;
    }

    static Set<Double> getAllValuesOfFeature(String key)
    {
        return allKeys.get(key);
    }


}


