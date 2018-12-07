package com.ML;

import javafx.util.Pair;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

class Probabilities {

    private HashMap<Pair<String, Double>, Double> weights;
    private HashMap<Pair<String, Double>, Double> negWeights;

    private double prior;

    Probabilities() {
        weights = new HashMap<>();
        prior = ThreadLocalRandom.current().nextDouble(0.0, 0.01);
    }

    double get(String feature, double value) {
        Pair key = new Pair(feature, value);

        if (!this.hasKey(key))
            this.put(key, ThreadLocalRandom.current().nextDouble(0.0, 0.01));
        return weights.get(key);
    }

    void put(String feature, double value, double probability) {
        weights.put(new Pair(feature, value), probability);
    }

    void put(Pair key, double probability) {
        weights.put(key, probability);
    }

    boolean hasKey(Pair key) {
        return weights.containsKey(key);
    }

    /**
     * Returns all features stored, with any value.
     * @return
     */
    Set<String> getAllKeys() {
        HashSet<String> ret = new HashSet<>();

        for(Pair eachPair : weights.keySet())
        {
            String eachKey = (String)(eachPair.getKey());
            ret.add(eachKey);
        }
        return ret;
    }

    public void setPrior(double _prior)
    {
        prior = _prior;
    }

    public double getPrior()
    {
        return prior;
    }


//    Weight copy()
//    {
//        Weight ret = new Weight();
//
//        ret.setB(this.getB());
//
//        for (String key : this.getAllKeys())
//        {
//            ret.put(key, this.get(key));
//        }
//
//        return ret;
//    }
//
//    void add(String key, double value)
//    {
//        if(!this.hasKey(key))
//            this.put(key, value);
//        else
//            this.put(key, this.get(key) + value);
//    }

}
