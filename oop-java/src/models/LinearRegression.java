//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: Linear Regression.java
//DATE: 11/14/2025
//--------------------------------------------------------
package models;

public class LinearRegression implements Model {
    
    private double[] weights;
    private double bias;
    private double l2;
    
    
    //Constructor
    public LinearRegression(double l2) {
        this.l2 = l2;
    }
    
    //Train the model using the closed-form normal equation 
    @Override
    public void fit(double[][] X, double[] y) {
        int n = X.length;        //# of samples
        int d = X[0].length;     //#of features
        
        // Add a bias column of 1's to X
        double[][] X_bias = addBiasColumn(X);
        
        // Compute X^T
        double[][] X_T = transpose(X_bias);
        
        // Compute X^T * X
        double[][] XTX = matrixMultiply(X_T, X_bias);
        
        // Add the L2 regularization: X^T X + lambda * I
        double epsilon = 1e-8;
        double regularization = Math.max(l2, epsilon);

        for (int i = 1; i < XTX.length; i++) {  // Start at 1 to skip the bias
            XTX[i][i] += regularization;
        }
        
        // Compute (X^T X)^-1
        double[][] XTX_inv = inverse(XTX);
        
        // Compute X^T * y
        double[] XTy = matrixVectorMultiply(X_T, y);
        
        // Compute w = (X^T X)^-1 * X^T * y
        double[] w_with_bias = matrixVectorMultiply(XTX_inv, XTy);
        
        // Extract the bias and the weights
        bias = w_with_bias[0];
        weights = new double[d];
        for (int i = 0; i < d; i++) {
            weights[i] = w_with_bias[i + 1];
        }
    }
    
    //Make predictions
    @Override
    public double[] predict(double[][] X) {
        int n = X.length;
        double[] predictions = new double[n];
        
        for (int i = 0; i < n; i++) {
            predictions[i] = bias;
            for (int j = 0; j < weights.length; j++) {
                predictions[i] += weights[j] * X[i][j];
            }
        }
        
        return predictions;
    }
    
    //Compute R^2 score
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        // Compute the mean of y
        double y_mean = 0.0;
        for (double val : y) {
            y_mean += val;
        }
        y_mean /= y.length;
        
        // Compute the SS_total and the SS_residual
        double ss_total = 0.0;
        double ss_residual = 0.0;
        
        for (int i = 0; i < y.length; i++) {
            ss_total += Math.pow(y[i] - y_mean, 2);
            ss_residual += Math.pow(y[i] - predictions[i], 2);
        }
        
        // R^2 = 1 - (SS_residual / SS_total)
        return 1.0 - (ss_residual / ss_total);
    }
    
    //Add a column of 1s to the left of X
    private double[][] addBiasColumn(double[][] X) {
        int n = X.length;
        int d = X[0].length;
        double[][] X_bias = new double[n][d + 1];
        
        for (int i = 0; i < n; i++) {
            X_bias[i][0] = 1.0;
            for (int j = 0; j < d; j++) {
                X_bias[i][j + 1] = X[i][j];
            }
        }
        return X_bias;
    }
    
    //Matrix transpose
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    //Matrix multiplication: A * B
    private double[][] matrixMultiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    //Matrix-vector multiplication: A * v
    private double[] matrixVectorMultiply(double[][] A, double[] v) {
        int rows = A.length;
        int cols = A[0].length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += A[i][j] * v[j];
            }
        }
        return result;
    }
    
    //Matrix inverse
    private double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create the augmented matrix [A | I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0;
    }
    
    for (int i = 0; i < n; i++) {
        // Find the largest element in the column
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        
        if (maxRow != i) {
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
        }
        
        double pivot = augmented[i][i];
        
        if (Math.abs(pivot) < 1e-10) {
            System.out.println("Warning: Nearly singular matrix detected. Adding regularization...");
            augmented[i][i] += 1e-8;
            pivot = augmented[i][i];
        }
        
        // Divide the row by pivot
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        
        // Get rid of the column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    
    // Extract the inverse
    double[][] inverse = new double[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = augmented[i][j + n];
        }
    }
    
    return inverse;
}
    
    // Getters
    public double[] getWeights() { return weights; }
    public double getBias() { return bias; }
}
