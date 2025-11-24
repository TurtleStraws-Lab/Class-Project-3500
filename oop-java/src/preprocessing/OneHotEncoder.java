//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: OneHotEncoder.java
//DATE: 11/7/2025
//--------------------------------------------------------
package preprocessing;

import java.util.*;

//OneHotEncoder
public class OneHotEncoder {
    
    private Map<String, Map<String, Integer>> encodingMaps;
    private Map<String, List<String>> uniqueValues;
    private String[] originalHeaders;
    private List<String> encodedHeaders;
    
    public OneHotEncoder() {
        encodingMaps = new HashMap<>();
        uniqueValues = new HashMap<>();
        encodedHeaders = new ArrayList<>();
    }
    
    //Fit the encoder on training data
    public void fit(String[] headers, List<String[]> trainData, List<Integer> categoricalIndices) {
        this.originalHeaders = headers;
        
        for (int colIdx : categoricalIndices) {
            String columnName = headers[colIdx];
            Set<String> uniqueSet = new HashSet<>();
            
            for (String[] row : trainData) {
                uniqueSet.add(row[colIdx]);
            }
            
            List<String> uniqueList = new ArrayList<>(uniqueSet);
            Collections.sort(uniqueList);
            uniqueValues.put(columnName, uniqueList);
            
            // Create encoding map
            Map<String, Integer> valueMap = new HashMap<>();
            for (int i = 0; i < uniqueList.size(); i++) {
                valueMap.put(uniqueList.get(i), i);
            }
            encodingMaps.put(columnName, valueMap);
        }
        
        // Build new headers
        for (int i = 0; i < headers.length; i++) {
            if (categoricalIndices.contains(i)) {
                String columnName = headers[i];
                List<String> values = uniqueValues.get(columnName);
                for (String value : values) {
                    encodedHeaders.add(columnName + "_" + value);
                }
            } else {
                encodedHeaders.add(headers[i]);
            }
        }
        
        //System.out.println("One-Hot Encoding fitted:");
        //System.out.println("  Original columns: " + headers.length);
        //System.out.println("  Encoded columns: " + encodedHeaders.size());
    }
    
    // Transform the data using the fitted encoder
    public double[][] transform(List<String[]> data, List<Integer> categoricalIndices) {
        int numRows = data.size();
        int numCols = encodedHeaders.size();
        double[][] transformed = new double[numRows][numCols];
        
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
            String[] row = data.get(rowIdx);
            int outputCol = 0;
            
            for (int colIdx = 0; colIdx < row.length; colIdx++) {
                if (categoricalIndices.contains(colIdx)) {
                    String columnName = originalHeaders[colIdx];
                    String value = row[colIdx];
                    List<String> values = uniqueValues.get(columnName);
                    
                    // Create a one-hot vector
                    for (String possibleValue : values) {
                        transformed[rowIdx][outputCol++] = value.equals(possibleValue) ? 1.0 : 0.0;
                    }
                } else {
                    // Keep the numeric value
                    try {
                        transformed[rowIdx][outputCol++] = Double.parseDouble(row[colIdx]);
                    } catch (NumberFormatException e) {
                        transformed[rowIdx][outputCol++] = 0.0;
                    }
                }
            }
        }
        
        return transformed;
    }
    
    //Fit and transform
    public double[][] fitTransform(String[] headers, List<String[]> data, List<Integer> categoricalIndices) {
        fit(headers, data, categoricalIndices);
        return transform(data, categoricalIndices);
    }
    
    //Get the encoded column headers
    public List<String> getEncodedHeaders() {
        return encodedHeaders;
    }
    
    //Get number of encoded columns
    public int getNumEncodedColumns() {
        return encodedHeaders.size();
    }
}
