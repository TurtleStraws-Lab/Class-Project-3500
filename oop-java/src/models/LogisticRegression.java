//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: LogisticRegression.java
//DATE: 11/20/2025
//--------------------------------------------------------
package models;

import java.util.Random;


public class LogisticRegression implements Model {
    
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private double l2;
    private Random random;
    
    //Constructor
    public LogisticRegression(double learningRate, int epochs, double l2, int seed) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.l2 = l2;
        this.random = new Random(seed);
    }
    
    //Train the model using gradient descent
    @Override
    public void fit(double[][] X, double[] y) {
        int n = X.length;        //#of samples
        int d = X[0].length;     //# of features
        
        // Initialize the weights randomly
        weights = new double[d];
        for (int i = 0; i < d; i++) {
            weights[i] = (random.nextDouble() - 0.5) * 0.01;  // Small random values
        }
        bias = 0.0;
        
        // The gradient descent
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Compute the predictions
            double[] predictions = new double[n];
            for (int i = 0; i < n; i++) {
                double z = bias;
                for (int j = 0; j < d; j++) {
                    z += weights[j] * X[i][j];
                }
                predictions[i] = sigmoid(z);
            }
            
            // Compute the gradients
            double[] dw = new double[d];
            double db = 0.0;
            
            for (int i = 0; i < n; i++) {
                double error = predictions[i] - y[i];
                db += error;
                for (int j = 0; j < d; j++) {
                    dw[j] += error * X[i][j];
                }
            }
            
            db /= n;
            for (int j = 0; j < d; j++) {
                dw[j] /= n;
                // Add the L2 regularization gradient
                if (l2 > 0) {
                    dw[j] += (l2 / n) * weights[j];
                }
            }
            
            // Weights and bias
            bias -= learningRate * db;
            for (int j = 0; j < d; j++) {
                weights[j] -= learningRate * dw[j];
            }
            
            if ((epoch + 1) % 100 == 0) {
                double loss = computeLoss(X, y, predictions);
                // System.out.println("Epoch " + (epoch + 1) + ", Loss: " + String.format("%.4f", loss));
            }
        }
    }
    
    //Make predictions (probabilities)
    public double[] predictProba(double[][] X) {
        int n = X.length;
        double[] probabilities = new double[n];
        
        for (int i = 0; i < n; i++) {
            double z = bias;
            for (int j = 0; j < weights.length; j++) {
                z += weights[j] * X[i][j];
            }
            probabilities[i] = sigmoid(z);
        }
        
        return probabilities;
    }
    
    // Make some predictions
    @Override
    public double[] predict(double[][] X) {
        double[] probabilities = predictProba(X);
        double[] predictions = new double[probabilities.length];
        
        for (int i = 0; i < probabilities.length; i++) {
            predictions[i] = probabilities[i] >= 0.5 ? 1.0 : 0.0;
        }
        
        return predictions;
    }
    
    //Compute accuracy score
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
    

    //Sigmoid activation function
    private double sigmoid(double z) {
        if (z > 20) return 1.0;
        if (z < -20) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    //Compute binary cross-entropy loss
    private double computeLoss(double[][] X, double[] y, double[] predictions) {
        int n = X.length;
        double loss = 0.0;
        
        for (int i = 0; i < n; i++) {
            double p = predictions[i];
            p = Math.max(1e-10, Math.min(1 - 1e-10, p));
            loss += -y[i] * Math.log(p) - (1 - y[i]) * Math.log(1 - p);
        }
        loss /= n;
        
        // Add L2 regularization
        if (l2 > 0) {
            double l2_term = 0.0;
            for (double w : weights) {
                l2_term += w * w;
            }
            loss += (l2 / (2 * n)) * l2_term;
        }
        
        return loss;
    }
    
    // Getters
    public double[] getWeights() { return weights; }
    public double getBias() { return bias; }
}
