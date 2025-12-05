//Name: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB - CMPS 3500
//FILE: Metrics.java
//DATE: 11/13/2025
//---------------------------------------------------

package metrics;

//metrics for regression and classification
public class Metrics {
    
    
     // Root Mean Squared Error, for regression
    public static double rmse(double[] y_true, double[] y_pred) {
        if (y_true.length != y_pred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        double sumSquaredError = 0.0;
        for (int i = 0; i < y_true.length; i++) {
            double error = y_true[i] - y_pred[i];
            sumSquaredError += error * error;
        }
        
        return Math.sqrt(sumSquaredError / y_true.length);
    }
    
    //R^2, the coefficient of determination, for regression
    public static double r2Score(double[] y_true, double[] y_pred) {
        if (y_true.length != y_pred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        // Compute the mean of y_true
        double y_mean = 0.0;
        for (double val : y_true) {
            y_mean += val;
        }
        y_mean /= y_true.length;
        
        // Make sure to compute the SS_total and SS_residual
        double ss_total = 0.0;
        double ss_residual = 0.0;
        
        for (int i = 0; i < y_true.length; i++) {
            ss_total += Math.pow(y_true[i] - y_mean, 2);
            ss_residual += Math.pow(y_true[i] - y_pred[i], 2);
        }
        
        // R^2 = 1 - (SS_residual / SS_total)
        return 1.0 - (ss_residual / ss_total);
    }
    
    
    //Accuracy - for classification
    public static double accuracy(double[] y_true, double[] y_pred) {
        if (y_true.length != y_pred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        int correct = 0;
        for (int i = 0; i < y_true.length; i++) {
            if (Math.abs(y_true[i] - y_pred[i]) < 0.5) {
                correct++;
            }
        }
        
        return (double) correct / y_true.length;
    }

    
    // Macro-F1 score - for binary
    public static double macroF1(double[] y_true, double[] y_pred) {
        if (y_true.length != y_pred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }

        // Count the TP, FP, FN for each class
        int tp0 = 0, fp0 = 0, fn0 = 0;
        int tp1 = 0, fp1 = 0, fn1 = 0;

        for (int i = 0; i < y_true.length; i++) {
            boolean trueIs0 = y_true[i] < 0.5;
            boolean predIs0 = y_pred[i] < 0.5;

            if (trueIs0 && predIs0) tp0++;
            else if (!trueIs0 && predIs0) fp0++;
            else if (trueIs0 && !predIs0) fn0++;

            if (!trueIs0 && !predIs0) tp1++;
            else if (trueIs0 && !predIs0) fp1++;
            else if (!trueIs0 && predIs0) fn1++;
        }

        // The calculation of  F1 for class 0
        double precision0 = (tp0 + fp0) > 0 ? (double) tp0 / (tp0 + fp0) : 0.0;
        double recall0 = (tp0 + fn0) > 0 ? (double) tp0 / (tp0 + fn0) : 0.0;
        double f1_0 = (precision0 + recall0) > 0 ? 2 * precision0 * recall0 / (precision0 + recall0) : 0.0;

        // The calculation of F1 for class 1
        double precision1 = (tp1 + fp1) > 0 ? (double) tp1 / (tp1 + fp1) : 0.0;
        double recall1 = (tp1 + fn1) > 0 ? (double) tp1 / (tp1 + fn1) : 0.0;
        double f1_1 = (precision1 + recall1) > 0 ? 2 * precision1 * recall1 / (precision1 + recall1) : 0.0;

        // Macro-F1 is the average  for the F1 scores
        return (f1_0 + f1_1) / 2.0;
    }
}
