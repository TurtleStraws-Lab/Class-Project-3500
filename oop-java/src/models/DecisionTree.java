//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: DecisionTree.java
//DATE: 11/25/2025
//--------------------------------------------------------

package models;

import java.util.*;

//Decision Tree (ID3)
public class DecisionTree implements Model {
    
    private TreeNode root;
    private int maxDepth;
    private int minSamplesSplit = 2;
    
    public DecisionTree(int maxDepth) {
        this.maxDepth = maxDepth;
    }
    
     //Train the decision tree
    @Override
    public void fit(double[][] X, double[] y) {
        root = buildTree(X, y, 0);
    }
    
    private TreeNode buildTree(double[][] X, double[] y, int depth) {
        int n = X.length;
        
        //The stopping conditions
        if (depth >= maxDepth || n < minSamplesSplit || isPure(y)) {
            return new TreeNode(majorityClass(y));
        }
        
        //Try to find the best split
        int bestFeature = findBestSplit(X, y);
        
        if (bestFeature == -1) {
            return new TreeNode(majorityClass(y));
        }
        
        // Create the node
        TreeNode node = new TreeNode(bestFeature);
        
        // Split data by the unique values of the best feature
        Map<Double, List<Integer>> splits = splitData(X, bestFeature);
        
        for (Map.Entry<Double, List<Integer>> entry : splits.entrySet()) {
            double value = entry.getKey();
            List<Integer> indices = entry.getValue();
            
            // Create the subset
            double[][] X_subset = new double[indices.size()][];
            double[] y_subset = new double[indices.size()];
            
            for (int i = 0; i < indices.size(); i++) {
                X_subset[i] = X[indices.get(i)];
                y_subset[i] = y[indices.get(i)];
            }
            
            // Recursively build the child
            TreeNode child = buildTree(X_subset, y_subset, depth + 1);
            node.children.put(value, child);
        }
        
        return node;
    }
    
    //Find the best feature to split
    private int findBestSplit(double[][] X, double[] y) {
        double baseEntropy = calculateEntropy(y);
        double bestGain = 0.0;
        int bestFeature = -1;
        
        int numFeatures = X[0].length;
        
        for (int feature = 0; feature < numFeatures; feature++) {
            double gain = informationGain(X, y, feature, baseEntropy);
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feature;
            }
        }
        
        return bestFeature;
    }
    
     //Calculate the information gain for the features
    private double informationGain(double[][] X, double[] y, int feature, double baseEntropy) {
        Map<Double, List<Integer>> splits = splitData(X, feature);
        
        double weightedEntropy = 0.0;
        int n = y.length;
        
        for (List<Integer> indices : splits.values()) {
            double[] subset_y = new double[indices.size()];
            for (int i = 0; i < indices.size(); i++) {
                subset_y[i] = y[indices.get(i)];
            }
            
            double weight = (double) indices.size() / n;
            weightedEntropy += weight * calculateEntropy(subset_y);
        }
        
        return baseEntropy - weightedEntropy;
    }
    
     //Split data by the unique values of a feature
    private Map<Double, List<Integer>> splitData(double[][] X, int feature) {
        Map<Double, List<Integer>> splits = new HashMap<>();
        
        for (int i = 0; i < X.length; i++) {
            double value = X[i][feature];
            splits.computeIfAbsent(value, k -> new ArrayList<>()).add(i);
        }
        
        return splits;
    }
    
     //Calculate the entropy of labels
    private double calculateEntropy(double[] y) {
        if (y.length == 0) return 0.0;
        
        // Count thr classes
        Map<Double, Integer> counts = new HashMap<>();
        for (double label : y) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        
        // Calculate the entropy
        double entropy = 0.0;
        int n = y.length;
        
        for (int count : counts.values()) {
            if (count > 0) {
                double p = (double) count / n;
                entropy -= p * Math.log(p) / Math.log(2);  // log base 2
            }
        }
        
        return entropy;
    }
    
     //Check if all  of the labels are the same
    private boolean isPure(double[] y) {
        if (y.length == 0) return true;
        
        double first = y[0];
        for (double label : y) {
            if (Math.abs(label - first) > 0.5) {
                return false;
            }
        }
        return true;
    }
    
     //Get the majority class from the  labels
    private double majorityClass(double[] y) {
        if (y.length == 0) return 0.0;
        
        Map<Double, Integer> counts = new HashMap<>();
        for (double label : y) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        
        double majorityLabel = 0.0;
        int maxCount = 0;
        
        for (Map.Entry<Double, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityLabel = entry.getKey();
            }
        }
        
        return majorityLabel;
    }
    
     //Make some predictions
    @Override
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingle(X[i], root);
        }
        
        return predictions;
    }
    
     //Predict a single sample by traversing the tree
    private double predictSingle(double[] x, TreeNode node) {
        // The leaf node
        if (node.isLeaf) {
            return node.label;
        }
        
        // Get the feature value
        double featureValue = x[node.featureIndex];
        
        // Traverse to child
        if (node.children.containsKey(featureValue)) {
            return predictSingle(x, node.children.get(featureValue));
        } else {
            return node.label;  // Use majority class
        }
    }
    
     //Calculate the accuracy
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (Math.abs(predictions[i] - y[i]) < 0.5) {
                correct++;
            }
        }
        
        return (double) correct / y.length;
    }
    
     //The inner class will  representing a tree node
    private static class TreeNode {
        boolean isLeaf;
        double label;
        int featureIndex;
        Map<Double, TreeNode> children;
        
        // The leaf node constructor
        TreeNode(double label) {
            this.isLeaf = true;
            this.label = label;
            this.children = new HashMap<>();
        }
        
        // The internal node constructor
        TreeNode(int featureIndex) {
            this.isLeaf = false;
            this.featureIndex = featureIndex;
            this.children = new HashMap<>();
            this.label = 0.0;
        }
    }
}
