package com.ML;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

class GeneralUtil {


    /**
     * Returns an ArrayList full of the examples from all given files
     *
     * @param files the files to read for the Examples
     * @return an ArrayList full of the examples from all given files
     */
    static ArrayList<Example> readExamples(String[] files, boolean presence) {
        ArrayList<Example> ret = new ArrayList<>();

        for (String file : files) {
            ArrayList<Example> part = readExamples(file, presence);
            ret.addAll(part);
        }

        return ret;
    }
    static ArrayList<Example> readExamples(String[] files) {
        return readExamples(files, false);
    }


    /**
     * Creates an ArrayList full of examples, as read in from a given file.
     *
     * @param fileName where the Example lines are read from
     * @return an ArrayList of read examples
     */
    static ArrayList<Example> readExamples(String fileName, boolean presence) {
        ArrayList<Example> ret = new ArrayList<>();

        BufferedReader reader = null;
        String line;


        try {
            reader = new BufferedReader(new FileReader(fileName));

            while ((line = reader.readLine()) != null) {
                Example next = new Example(line, presence);
                ret.add(next);
            }


        } catch (
                FileNotFoundException e) {
            System.out.println("File " + fileName + " not found.");
        } catch (
                IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        return ret;
    }
    static ArrayList<Example> readExamples(String fileName) {
        return readExamples(fileName, false);
    }

    static Double smallRandom() {
        return ThreadLocalRandom.current().nextDouble(0.0, 0.01);
    }



}
