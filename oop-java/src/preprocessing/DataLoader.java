//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: DataLoader.java
//DATE: 11/6/2025
//--------------------------------------------------------
package preprocessing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//DataLoader - Handles CSV file loading
public class DataLoader {
    
    private String[] headers;
    private List<String[]> rows;
    private int numRows;
    private int numCols;
    
    //Load the CSV file from the path
    public void loadCSV(String filePath) throws IOException {
        long startTime = System.currentTimeMillis();
        
        System.out.println("[" + new java.util.Date() + "] Starting Script");
    
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        rows = new ArrayList<>();
    
        System.out.println("[" + new java.util.Date() + "] Loading training data set");
        
        // Reads the header line
        String headerLine = reader.readLine();
        if (headerLine == null) {
            reader.close();
            throw new IOException("CSV file is empty");
        }
        
        headers = headerLine.split(",");
        numCols = headers.length;
        
        for (int i = 0; i < headers.length; i++) {
            headers[i] = headers[i].trim();
        }
        
        System.out.println("[" + new java.util.Date() + "] Total Columns Read: " + numCols);
        
        // Reads from the data rows
        String line;
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            
            for (int i = 0; i < values.length; i++) {
                values[i] = values[i].trim();
            }
            
            rows.add(values);
        }
        
        reader.close();
        numRows = rows.size();
        
        System.out.println("[" + new java.util.Date() + "] Total Rows Read: " + numRows);
    
        long endTime = System.currentTimeMillis();
        System.out.println("\nTime to load is: " + (endTime - startTime) / 1000.0 + " seconds");
    }

    public String[] getHeaders() {
        return headers;
    }
    
    //Get all data rows
    public List<String[]> getRows() {
        return rows;
    }
    
    //Get the number of rows loaded
    public int getNumRows() {
        return numRows;
    }
    
    //Get the number of columns
    public int getNumCols() {
        return numCols;
    }
    
    //Get the index of a column by name
    public int getColumnIndex(String columnName) {
        for (int i = 0; i < headers.length; i++) {
            if (headers[i].equalsIgnoreCase(columnName)) {
                return i;
            }
        }
        return -1;
    }
    
    //Extracts a column as a String array
    public String[] getColumn(String columnName) {
        int colIndex = getColumnIndex(columnName);
        if (colIndex == -1) {
            System.err.println("Column '" + columnName + "' not found!");
            return null;
        }
        
        String[] column = new String[numRows];
        for (int i = 0; i < numRows; i++) {
            column[i] = rows.get(i)[colIndex];
        }
        return column;
    }
    
    //Extracts a column as a double array
    public double[] getNumericColumn(String columnName) {
        String[] strColumn = getColumn(columnName);
        if (strColumn == null) {
            return null;
        }
        
        double[] numColumn = new double[strColumn.length];
        for (int i = 0; i < strColumn.length; i++) {
            try {
                numColumn[i] = Double.parseDouble(strColumn[i]);
            } catch (NumberFormatException e) {
                System.err.println("Warning: Could not parse '" + strColumn[i] + "' as number. Using 0.0");
                numColumn[i] = 0.0;
            }
        }
        return numColumn;
    }
    
    //Checks if a column contains numeric data
    public boolean isNumericColumn(String columnName) {
        String[] column = getColumn(columnName);
        if (column == null || column.length == 0) {
            return false;
        }
        
        // Checks the first few values
        int samplesToCheck = Math.min(10, column.length);
        for (int i = 0; i < samplesToCheck; i++) {
            try {
                Double.parseDouble(column[i]);
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }
    
    //Get all column names that are numeric
    public List<String> getNumericColumns() {
        List<String> numericCols = new ArrayList<>();
        for (String header : headers) {
            if (isNumericColumn(header)) {
                numericCols.add(header);
            }
        }
        return numericCols;
    }
    
    //Gets all of the column names that are categorical
    public List<String> getCategoricalColumns() {
        List<String> categoricalCols = new ArrayList<>();
        for (String header : headers) {
            if (!isNumericColumn(header)) {
                categoricalCols.add(header);
            }
        }
        return categoricalCols;
    }
    
    //Prints a summary of the loaded data
    public void printSummary() {
        System.out.println("\n=== Data Summary ===");
        System.out.println("Rows: " + numRows);
        System.out.println("Columns: " + numCols);
        System.out.println("\nColumn Names:");
        for (int i = 0; i < headers.length; i++) {
            String type = isNumericColumn(headers[i]) ? "numeric" : "categorical";
            System.out.println("  " + (i+1) + ". " + headers[i] + " (" + type + ")");
        }
        System.out.println("===================\n");
    }
}
