//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: Preprocessor.java
//DATE: 11/8/2025
//--------------------------------------------------------
package preprocessing;

import java.util.ArrayList;
import java.util.List;

//Preprocessor
public class Preprocessor {
    
    private DataLoader dataLoader;
    private TrainTestSplitter splitter;
    private OneHotEncoder encoder;
    private Normalizer normalizer;
    
    private double[][] X_train;
    private double[][] X_test;
    private double[] y_train;
    private double[] y_test;
    
    private String targetColumn;
    
    public Preprocessor(DataLoader dataLoader) {
        if (dataLoader == null) {
            throw new IllegalArgumentException("DataLoader cannot be null");
        }
        this.dataLoader = dataLoader;
    }
    
    // The preprocessing pipeline
    public void preprocess(String targetColumn, double trainRatio, long seed, boolean normalize) {
        this.targetColumn = targetColumn;
        
        //System.out.println("\n=== Starting Preprocessing ===");
        
        // Verify that the  data is loaded
        if (dataLoader.getRows() == null || dataLoader.getRows().isEmpty()) {
            throw new IllegalStateException("No data loaded in DataLoader");
        }
        
        //System.out.println("Total rows to process: " + dataLoader.getRows().size());
        
        // Train/Test Split
        //System.out.println("Performing train/test split...");
        splitter = new TrainTestSplitter(seed);
        splitter.split(dataLoader.getRows(), trainRatio);
        
        // Step 2: Identify target column index
        int targetIndex = dataLoader.getColumnIndex(targetColumn);
        if (targetIndex == -1) {
            throw new IllegalArgumentException("Target column '" + targetColumn + "' not found");
        }
        //System.out.println("Target column '" + targetColumn + "' found at index " + targetIndex);
        
        //System.out.println("Extracting target variable...");
        y_train = extractTarget(splitter.getTrainData(), targetIndex);
        y_test = extractTarget(splitter.getTestData(), targetIndex);
        
        //System.out.println("Target variable extracted: " + targetColumn);
        //System.out.println("  Unique train values: " + countUnique(y_train));
        //System.out.println("  Unique test values: " + countUnique(y_test));
        
        //System.out.println("Identifying categorical columns...");
        List<Integer> categoricalIndices = new ArrayList<>();
        String[] headers = dataLoader.getHeaders();
        for (int i = 0; i < headers.length; i++) {
            if (i != targetIndex && !dataLoader.isNumericColumn(headers[i])) {
                categoricalIndices.add(i);
            }
        }
        //System.out.println("Found " + categoricalIndices.size() + " categorical columns");
        
        //System.out.println("Removing target column from features...");
        List<String[]> trainDataNoTarget = removeColumn(splitter.getTrainData(), targetIndex);
        List<String[]> testDataNoTarget = removeColumn(splitter.getTestData(), targetIndex);
        String[] headersNoTarget = removeHeader(headers, targetIndex);
        
        // Adjust categorical indices after removing target column
        List<Integer> adjustedCategoricalIndices = adjustIndices(categoricalIndices, targetIndex);
        
        //System.out.println("Performing one-hot encoding...");
        encoder = new OneHotEncoder();
        X_train = encoder.fitTransform(headersNoTarget, trainDataNoTarget, adjustedCategoricalIndices);
        X_test = encoder.transform(testDataNoTarget, adjustedCategoricalIndices);
        
        if (normalize) {
            //System.out.println("Applying z-score normalization...");
            normalizer = new Normalizer();
            X_train = normalizer.fitTransform(X_train);
            X_test = normalizer.transform(X_test);
        }
        
        //System.out.println("\n=== Preprocessing Complete ===");
        //System.out.println("Training set: " + X_train.length + " samples" + X_train[0].length + " features");
        //System.out.println("Test set: " + X_test.length + " samples " + X_test[0].length + " features");
        //System.out.println("==============================\n");
    }
    
    
    // Extract the target variable and convert to numeric if needed
    private double[] extractTarget(List<String[]> data, int targetIndex) {
        double[] target = new double[data.size()];
        
        // Check if the target is already numeric
        boolean isNumeric = true;
        try {
            Double.parseDouble(data.get(0)[targetIndex]);
        } catch (NumberFormatException e) {
            isNumeric = false;
        }
        
        if (isNumeric) {
            for (int i = 0; i < data.size(); i++) {
                target[i] = Double.parseDouble(data.get(i)[targetIndex]);
            }
        } else {
            for (int i = 0; i < data.size(); i++) {
                String value = data.get(i)[targetIndex];
                target[i] = value.contains(">") ? 1.0 : 0.0;
            }
        }
        
        return target;
    }
    
    //Remove a column from data
    private List<String[]> removeColumn(List<String[]> data, int columnIndex) {
        List<String[]> result = new ArrayList<>();
        for (String[] row : data) {
            String[] newRow = new String[row.length - 1];
            int idx = 0;
            for (int i = 0; i < row.length; i++) {
                if (i != columnIndex) {
                    newRow[idx++] = row[i];
                }
            }
            result.add(newRow);
        }
        return result;
    }
    
    //Remove a header from headers array
    private String[] removeHeader(String[] headers, int columnIndex) {
        String[] newHeaders = new String[headers.length - 1];
        int idx = 0;
        for (int i = 0; i < headers.length; i++) {
            if (i != columnIndex) {
                newHeaders[idx++] = headers[i];
            }
        }
        return newHeaders;
    }
    
    //Adjust indices after removing a colum
    private List<Integer> adjustIndices(List<Integer> indices, int removedIndex) {
        List<Integer> adjusted = new ArrayList<>();
        for (int idx : indices) {
            if (idx < removedIndex) {
                adjusted.add(idx);
            } else if (idx > removedIndex) {
                adjusted.add(idx - 1);
            }
        }
        return adjusted;
    }
    
    // Count the unique values in array
    private int countUnique(double[] arr) {
        java.util.Set<Double> unique = new java.util.HashSet<>();
        for (double val : arr) {
            unique.add(val);
        }
        return unique.size();
    }
    
    // Getters
    public double[][] getXTrain() { return X_train; }
    public double[][] getXTest() { return X_test; }
    public double[] getYTrain() { return y_train; }
    public double[] getYTest() { return y_test; }
}
