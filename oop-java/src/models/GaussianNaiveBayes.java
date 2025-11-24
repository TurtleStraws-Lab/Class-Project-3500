//Name: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB - CMPS 3500
//FILE: GaussianNaiveBayes.java
//DATE: 11/23/2025
//---------------------------------------------------
package models;

import java.util.*;

//Gaussian Naive Bayes
public class GaussianNaiveBayes implements Model {
    
    private double[] classPriors;
    private double[][] means;
    private double[][] variances;
    private double[] classes;
    private int numFeatures;
    private double varianceSmoothing = 1e-3;
    
    //Constructor
    public GaussianNaiveBayes() {
    }
    
    
     //Train the model
    @Override
    public void fit(double[][] X, double[] y) {
        int n = X.length;
        numFeatures = X[0].length;
        
        // Find the unique classes
        Set<Double> classSet = new HashSet<>();
        for (double label : y) {
            classSet.add(label);
        }
        classes = new double[classSet.size()];
        int idx = 0;
        for (double c : classSet) {
            classes[idx++] = c;
        }
        Arrays.sort(classes);  // Sort for consistency
        
        int numClasses = classes.length;
        
        // Initialize the arrays
        classPriors = new double[numClasses];
        means = new double[numClasses][numFeatures];
        variances = new double[numClasses][numFeatures];
        
        // Compute the statistics for each class
        for (int c = 0; c < numClasses; c++) {
            double classLabel = classes[c];
            
            // Try to find the  samples belonging to this class
            List<double[]> classSamples = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (Math.abs(y[i] - classLabel) < 0.5) {
                    classSamples.add(X[i]);
                }
            }
            
            int classCount = classSamples.size();
            
            if (classCount == 0) {
                throw new IllegalStateException("No samples found for class " + classLabel);
            }
            
            // Compute the class prior
            classPriors[c] = (double) classCount / n;
            
            // Compute the mean for each feature
            for (int j = 0; j < numFeatures; j++) {
                double sum = 0.0;
                for (double[] sample : classSamples) {
                    sum += sample[j];
                }
                means[c][j] = sum / classCount;
            }
            
            // Compute the variance for each feature
            for (int j = 0; j < numFeatures; j++) {
                double sumSquaredDiff = 0.0;
                for (double[] sample : classSamples) {
                    double diff = sample[j] - means[c][j];
                    sumSquaredDiff += diff * diff;
                }
                
                // Use the variance calculation
                if (classCount > 1) {
                    variances[c][j] = sumSquaredDiff / classCount;
                } else {
                    variances[c][j] = 1.0;
                }
                
                // Ensure the  minimum variance
                if (variances[c][j] < varianceSmoothing) {
                    variances[c][j] = varianceSmoothing;
                }
            }
        }
        
        //Print the class distribution
        System.out.println("Class distribution:");
        for (int c = 0; c < numClasses; c++) {
            System.out.println("  Class " + classes[c] + ": prior = " + String.format("%.4f", classPriors[c]));
        }
    }
    
    //Make predictions
    @Override
    public double[] predict(double[][] X) {
        int n = X.length;
        double[] predictions = new double[n];
        
        for (int i = 0; i < n; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        
        return predictions;
    }
    
     //Predict class for a single sample
    private double predictSingle(double[] x) {
        int numClasses = classes.length;
        double[] logProbabilities = new double[numClasses];
        
        // Compute the log probability for each class
        for (int c = 0; c < numClasses; c++) {
            // Make sure to start with log for each class prior
            logProbabilities[c] = Math.log(classPriors[c]);
            
            for (int j = 0; j < numFeatures; j++) {
                logProbabilities[c] += logGaussianProbability(x[j], means[c][j], variances[c][j]);
            }
        }
        
        // Return the class with the highest log probability
        int maxIndex = 0;
        for (int c = 1; c < numClasses; c++) {
            if (logProbabilities[c] > logProbabilities[maxIndex]) {
                maxIndex = c;
            }
        }
        
        return classes[maxIndex];
    }
    
     //Compute the log of the density function
    private double logGaussianProbability(double x, double mean, double variance) {
        // log P(x|mean,var) = -0.5 * log(2*var) - (x-mean)^2 / (2*var)
        double diff = x - mean;
        double logProb = -0.5 * Math.log(2 * Math.PI * variance);
        logProb -= (diff * diff) / (2 * variance);
        return logProb;
    }
    
    // Compute the accuracy score
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
    
    // These are the getters
    public double[] getClassPriors() { return classPriors; }
    public double[][] getMeans() { return means; }
    public double[][] getVariances() { return variances; }
}
