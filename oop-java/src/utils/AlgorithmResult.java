//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: AlgorithmResult.java
//DATE: 11/26/2025
//--------------------------------------------------------
package utils;

//AlgorithmResult - Stores results from a single algorithm run
public class AlgorithmResult {
    private String implementation;
    private String algorithm;
    private double trainTime;
    private double metric1;
    private String metric1Name;
    private double metric2;
    private String metric2Name;
    private int sloc;
    
    public AlgorithmResult(String implementation, String algorithm, double trainTime,
                          String metric1Name, double metric1,
                          String metric2Name, double metric2, int sloc) {
        this.implementation = implementation;
        this.algorithm = algorithm;
        this.trainTime = trainTime;
        this.metric1Name = metric1Name;
        this.metric1 = metric1;
        this.metric2Name = metric2Name;
        this.metric2 = metric2;
        this.sloc = sloc;
    }
    
    // Getters
    public String getImplementation() { return implementation; }
    public String getAlgorithm() { return algorithm; }
    public double getTrainTime() { return trainTime; }
    public String getMetric1Name() { return metric1Name; }
    public double getMetric1() { return metric1; }
    public String getMetric2Name() { return metric2Name; }
    public double getMetric2() { return metric2; }
    public int getSloc() { return sloc; }
}
