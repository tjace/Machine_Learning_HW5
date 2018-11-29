package com.ML;

import java.util.HashMap;
import java.util.Set;

/**
 * an Example's features array contains:
 * features[0]:         -1.0 or 1.0, corresponding with the label
 * features [1 - 19]:   a double, corresponding to the fields 1 - 19.  1 is the lowest field, and 19 is the highest.
 */
class Example {

    private boolean label;
    private HashMap<String, Double> features;

    Example(String fullLine) {
        String[] pieces = fullLine.split(" ");
        features = new HashMap<>();

        for (String each : pieces) {
            switch (each) {
                case "-1":
                case "0":
                    label = false;
                    break;
                case "+1":
                case "1":
                    label = true;
                    break;
                default:
                    String[] splits = each.split(":");
                    features.put(splits[0], Double.parseDouble(splits[1]));
            }
        }
    }


    Double get(String n) {
        return features.get(n);
    }

    boolean getLabel() {
        return label;
    }

    boolean hasKey(String key) {
        return features.containsKey(key);
    }

    Set<String> getAllKeys() {
        return features.keySet();

    }
}


