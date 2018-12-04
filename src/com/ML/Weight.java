package com.ML;

import java.util.HashMap;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

class Weight {

    private HashMap<String, Double> weights;
    private double b;

    Weight() {
        weights = new HashMap<>();
        b = ThreadLocalRandom.current().nextDouble(0.0, 0.01);
    }

    double get(String key) {
        if (!this.hasKey(key))
            this.put(key, ThreadLocalRandom.current().nextDouble(0.0, 0.01));
        return weights.get(key);
    }

    void put(String key, double value) {
        weights.put(key, value);
    }

    boolean hasKey(String key) {
        return weights.containsKey(key);
    }

    double getB() {
        return b;
    }

    void setB(Double num) {
        b = num;
    }

    Set<String> getAllKeys() {
        return weights.keySet();
    }

    Weight copy()
    {
        Weight ret = new Weight();

        ret.setB(this.getB());

        for (String key : this.getAllKeys())
        {
            ret.put(key, this.get(key));
        }

        return ret;
    }

    void add(String key, double value)
    {
        if(!this.hasKey(key))
            this.put(key, value);
        else
            this.put(key, this.get(key) + value);
    }

}
