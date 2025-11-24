//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: Normalizer.java
//DATE: 11/7/2025
//--------------------------------------------------------
package preprocessing;

//Normalizer - Z-score normalization for numeric features
public class Normalizer {
    
    private double[] means;
    private double[] stds;
    private int numFeatures;
    
    //Fit the normalizer on training data, compute the  means and standard deviations)
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }
        
        int numSamples = X.length;
        numFeatures = X[0].length;
        
        means = new double[numFeatures];
        stds = new double[numFeatures];
        
        // Computes the means
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < numSamples; i++) {
                sum += X[i][j];
            }
            means[j] = sum / numSamples;
        }
        
        // Computes the standard deviations
        for (int j = 0; j < numFeatures; j++) {
            double sumSquaredDiff = 0.0;
            for (int i = 0; i < numSamples; i++) {
                double diff = X[i][j] - means[j];
                sumSquaredDiff += diff * diff;
            }
            stds[j] = Math.sqrt(sumSquaredDiff / numSamples);
            
            // Avoid division by zero
            if (stds[j] < 1e-10) {
                stds[j] = 1.0;
            }
        }
        
        //System.out.println("Normalization fitted on " + numFeatures + " features");
    }
    
    //Transform the data using fitted means and standard deviations
    public double[][] transform(double[][] X) {
        if (means == null || stds == null) {
            throw new IllegalStateException("Normalizer must be fitted before transform");
        }
        
        int numSamples = X.length;
        double[][] normalized = new double[numSamples][numFeatures];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                normalized[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }
        
        return normalized;
    }
    
    //Fit and transform
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }
    
    //Get the computed means
    public double[] getMeans() {
        return means;
    }
    
    //Get the computed standard deviation
    public double[] getStds() {
        return stds;
    }
}
