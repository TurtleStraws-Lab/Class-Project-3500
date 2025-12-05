//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: KNearestNeighbors.java
//DATE: 11/18/2025
//--------------------------------------------------------
package models;

import java.util.Arrays;
import java.util.Comparator;

//k-Nearest Neighbors - Distance-based classification
public class KNearestNeighbors implements Model {
    
    private int k;
    private double[][] X_train;
    private double[] y_train;
    
    //Constructor
    public KNearestNeighbors(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        this.k = k;
    }
    
    // Train the model
    @Override
    public void fit(double[][] X, double[] y) {
        // Store the data
        this.X_train = X;
        this.y_train = y;
    }
    
    //Make predictions for the test data
    @Override
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        
        return predictions;
    }
    
    //Predict labels for a test sample
    private double predictSingle(double[] x_test) {
        // Calculate the distances to the training points
        double[] distances = new double[X_train.length];
        
        for (int i = 0; i < X_train.length; i++) {
            distances[i] = euclideanDistance(x_test, X_train[i]);
        }
        
        // Find the indices for thje knn
        int[] neighborIndices = findKNearest(distances, k);
        
        // Get the labels of knn
        double[] neighborLabels = new double[k];
        for (int i = 0; i < k; i++) {
            neighborLabels[i] = y_train[neighborIndices[i]];
        }
        
        return majorityVote(neighborLabels);
    }
    
     //Calculate the euclidean distance between two points
    private double euclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        
        return Math.sqrt(sum);
    }
    
    //Find indices of k smallest distances
    private int[] findKNearest(double[] distances, int k) {
        // Make sure to create array of pairs
        IndexDistancePair[] pairs = new IndexDistancePair[distances.length];
        for (int i = 0; i < distances.length; i++) {
            pairs[i] = new IndexDistancePair(i, distances[i]);
        }
        
        // Sort by distance
        Arrays.sort(pairs, Comparator.comparingDouble(p -> p.distance));
        
        int[] kNearest = new int[k];
        for (int i = 0; i < k; i++) {
            kNearest[i] = pairs[i].index;
        }
        
        return kNearest;
    }
    
    //Get majority vote from neighbor labels
    private double majorityVote(double[] labels) {
        int count0 = 0;
        int count1 = 0;
        
        for (double label : labels) {
            if (label < 0.5) {
                count0++;
            } else {
                count1++;
            }
        }
        
        // Return majority class,howver if theres a tie, return 1
        return count1 >= count0 ? 1.0 : 0.0;
    }
    
    //Calculate the  accuracy on a test set
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

    //class stores the index-distance pairs
    private static class IndexDistancePair {
        int index;
        double distance;
        
        IndexDistancePair(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }
    }
}
