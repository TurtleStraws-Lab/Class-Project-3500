//NAME: Nicole Flanders
//ASGT: Class Project
//ORGN: CSUB- CMPS 3500
//FILE: Model.java
//DATE: 11/10/2025
//--------------------------------------------------------
package models;

//Base interface for all of the ML models
public interface Model {
    
    //Train the models on the given data
    void fit(double[][] X, double[] y);
    
    //Make predictions on the new data
    double[] predict(double[][] X);
    
    //Evaluate the model performance
    double score(double[][] X, double[] y);
}
