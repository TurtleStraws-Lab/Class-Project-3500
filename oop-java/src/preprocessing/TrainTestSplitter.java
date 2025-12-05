//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: TrainTestSplitter.java
//DATE: 11/8/2025
//--------------------------------------------------------
package preprocessing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

//TrainTestSplitter
public class TrainTestSplitter {
    
    private List<String[]> trainData;
    private List<String[]> testData;
    private Random random;
    
    //Constructor
    public TrainTestSplitter(long seed) {
        this.random = new Random(seed);
    }
    
    //Split data into train and test sets
    public void split(List<String[]> data, double trainRatio) {
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }
        
        // Create a copy and shuffle
        List<String[]> shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, random);
        
        // Calculate the split point
        int trainSize = (int) (shuffled.size() * trainRatio);
        
        // Split the data
        trainData = new ArrayList<>(shuffled.subList(0, trainSize));
        testData = new ArrayList<>(shuffled.subList(trainSize, shuffled.size()));
        
        //System.out.println("Train/Test Split:");
        //System.out.println("  Training samples: " + trainData.size());
        //System.out.println("  Testing samples: " + testData.size());
    }
    
    //Get training data
    public List<String[]> getTrainData() {
        return trainData;
    }
    
    //Get testing data
    public List<String[]> getTestData() {
        return testData;
    }
    
    //Extract a specific column from train data
    public static String[] extractColumn(List<String[]> data, int columnIndex) {
        String[] column = new String[data.size()];
        for (int i = 0; i < data.size(); i++) {
            column[i] = data.get(i)[columnIndex];
        }
        return column;
    }
    
    //Extract a numeric column as double array
    public static double[] extractNumericColumn(List<String[]> data, int columnIndex) {
        double[] column = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            try {
                column[i] = Double.parseDouble(data.get(i)[columnIndex]);
            } catch (NumberFormatException e) {
                column[i] = 0.0;
            }
        }
        return column;
    }
}
