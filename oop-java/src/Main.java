//NAME: Nicole Flanders
//ASGT: Class Project
//ORGIN: CSUB-CMPS 3500
//FILE: Main.java
//DATE: 11/10/2025
//------------------------------------------------------------------------------
import models.*;
import preprocessing.*;
import metrics.*;
import utils.*;
import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;


//  Main file for the ML Paradigms Project - Java OOP Implementation
public class Main {
    private static Scanner scanner = new Scanner(System.in);
    private static String dataPath = "../../data/adult_income_cleaned.csv";

    // These are the data objects
    private static DataLoader dataLoader = null;
    private static Preprocessor preprocessor = null;
    
    private static double[][] X_train = null;
    private static double[][] X_test = null;
    private static double[] y_train = null;
    private static double[] y_test = null;
    private static boolean dataLoaded = false;

    //Result storage
    private static List<AlgorithmResult> results = new ArrayList<>();
    
    public static void main(String[] args) {
        boolean running = true;
        
        while (running) {
            printMenu();
            int choice = getChoice();
            
            switch (choice) {
                case 1:
                    loadData();
                    break;
                case 2:
                    runLinearRegression();
                    break;
                case 3:
                    runLogisticRegression();
                    break;
                case 4:
                    runKNN();
                    break;
                case 5:
                    runDecisionTree();
                    break;
                case 6:
                    runNaiveBayes();
                    break;
                case 7:
                    printResults();
                    break;
                case 8:
                    running = false;
                    System.out.println("Goodbye!");
                    break;
                default:
                    System.out.println("Invalid option. Try again.");
            }
        }
        scanner.close();
    }
    
    private static void printMenu() {
        System.out.println("\n=== ML Paradigms Project - Java OOP ===");
        System.out.println("(1) Load data");
        System.out.println("(2) Linear Regression (closed-form)");
        System.out.println("(3) Logistic Regression (binary)");
        System.out.println("(4) k-Nearest Neighbors");
        System.out.println("(5) Decision Tree (ID3)");
        System.out.println("(6) Gaussian Naive Bayes");
        System.out.println("(7) Print general results");
        System.out.println("(8) Quit");
        System.out.print("\nEnter option: ");
    }
    
    private static int getChoice() {
        try {
            return Integer.parseInt(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }
    
    private static void loadData() {
        System.out.println("\nPick dataset from list:\n");
        System.out.println("  1. adult_income_cleaned.csv");
        System.out.println("  2. adult_income_demo0.csv");
        System.out.println("  3. adult_income_demo1.csv");
        System.out.println("  4. adult_income_demo2.csv");
        System.out.println("  5. adult_income_demo3.csv");
        System.out.println("  6. adult_income_demo4.csv");
        System.out.print("\nEnter an option: ");
        
        int choice = 1;
        try {
            choice = Integer.parseInt(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("Invalid input, using option 1");
        }

        // Map menu choices to file paths
        switch (choice) {
            case 1:
                dataPath = "../../data/adult_income_cleaned.csv";
                break;
            case 2:
                dataPath = "../../data/adult_income_demo0.csv";
                break;
            case 3:
                dataPath = "../../data/adult_income_demo1.csv";
                break;
            case 4:
                dataPath = "../../data/adult_income_demo2.csv";
                break;
            case 5:
                dataPath = "../../data/adult_income_demo3.csv";
                break;
            case 6:
                dataPath = "../../data/adult_income_demo4.csv";
                break;
            default:
                System.out.println("Invalid option, defaulting to option 1");
                dataPath = "../../data/adult_income_cleaned.csv";
    }

    System.out.println("\nLoading dataset: " + dataPath);
    System.out.println("************************************");
    
        
    
        try {
            //Load the CSV file,make sure its the clean one!
            dataLoader = new DataLoader();
            dataLoader.loadCSV(dataPath);
        
            System.out.println();
        
            // Preprocess for the classification
            preprocessor = new Preprocessor(dataLoader);
            preprocessor.preprocess("income", 0.8, 42, true);
        
            //Get preprocessed data
            X_train = preprocessor.getXTrain();
            X_test = preprocessor.getXTest();
            y_train = preprocessor.getYTrain();
            y_test = preprocessor.getYTest();
        
            dataLoaded = true;
        
        } catch (Exception e) {
            System.err.println("Error loading data: " + e.getMessage());
            e.printStackTrace();
            dataLoaded = false;
        }
    }
    
    private static void runLinearRegression() {
        if (!dataLoaded) {
            System.out.println("Please load data first (option 1)");
            return;
        }
        
        System.out.println("\nLinear Regression (closed-form):");
        System.out.println("********************************");
        System.out.println("Enter input options:\n");
    
        // Input option 1: the target variable
        System.out.println("Input option 1: Target variable: hours.per.week");
    
        // Input option 2: the L2 regularization
        System.out.print("Input option 2: L2 = ");
        double l2 = 0.0;
        try {
            String input = scanner.nextLine().trim();
            l2 = Double.parseDouble(input);
            System.out.println("  (L2 regularization strength set to " + l2 + ")");
        } catch (NumberFormatException e) {
            System.out.println("0 (no regularization)");
            l2 = 0.0;
        }
    
        try {
            Preprocessor regPreprocessor = new Preprocessor(dataLoader);
            regPreprocessor.preprocess("hours.per.week", 0.8, 42, true);
        
            double[][] X_train_reg = regPreprocessor.getXTrain();
            double[][] X_test_reg = regPreprocessor.getXTest();
            double[] y_train_reg = regPreprocessor.getYTrain();
            double[] y_test_reg = regPreprocessor.getYTest();
        
            // Train the model
            long startTime = System.currentTimeMillis();
        
            LinearRegression model = new LinearRegression(l2);
            model.fit(X_train_reg, y_train_reg);
        
            long endTime = System.currentTimeMillis();
            double trainTime = (endTime - startTime) / 1000.0;
        
            // Make predictions
            double[] y_pred_test = model.predict(X_test_reg);
        
            // Evaluate
            double rmse = Metrics.rmse(y_test_reg, y_pred_test);
            double r2 = Metrics.r2Score(y_test_reg, y_pred_test);
        
            // Count the SLOC
            int sloc = countSloc("models/LinearRegression.java");
        
            // Display the results
            System.out.println("\nOutputs:");
            System.out.println("*******");
            System.out.println("Algorithm: Linear Regression (closed-form)");
            System.out.println("Train time: " + String.format("%.3f", trainTime) + " seconds");
            System.out.println("Metric 1: RMSE: " + String.format("%.4f", rmse));
            System.out.println("Metric 2: R^2: " + String.format("%.4f", r2));
            System.out.println("Metric 3: SLOC: " + sloc);

            //Store the result
            results.add(new AlgorithmResult("Java", "Linear Regression", trainTime,
                                        "RMSE", rmse, "R^2", r2, sloc));

        } catch (Exception e) {
            System.err.println("Error during Linear Regression: " + e.getMessage());
            e.printStackTrace();
        }
    }
 
    
    private static void runLogisticRegression() {
        if (!dataLoaded) {
            System.out.println("Please load data first (option 1)");
            return;
        }
        
        System.out.println("\nLogistic Regression (binary):");
        System.out.println("*****************************");
        System.out.println("Enter input options:\n");

        System.out.println("Input option 1: Target variable: income");

    
        System.out.print("Input option 2: lr = ");
        double lr = 0.2;
        try {
            lr = Double.parseDouble(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("0.2 (default)");
        }
        System.out.println("     --> learning rate");

        System.out.print("Input option 3: epochs = ");
        int epochs = 400;
        try {
            epochs = Integer.parseInt(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("400 (default)");
        }
        System.out.println("     --> number of training epochs");

        System.out.print("Input option 4: l2 = ");
        double l2 = 0.003;
        try {
            l2 = Double.parseDouble(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("0.003 (default)");
        }
        System.out.println("     --> L2 regularization strength");

        System.out.print("Input option 5: seed = ");
        int seed = 7;
        try {
            seed = Integer.parseInt(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("7 (default)");
        }
        System.out.println("     --> reproducible initialization");

        try {
            // Train the model
            long startTime = System.currentTimeMillis();

            LogisticRegression model = new LogisticRegression(lr, epochs, l2, seed);
            model.fit(X_train, y_train);

            long endTime = System.currentTimeMillis();
            double trainTime = (endTime - startTime) / 1000.0;

            // Make some predictions
            double[] y_pred_test = model.predict(X_test);

            // Evaluate
            double accuracy = Metrics.accuracy(y_test, y_pred_test);
            double macroF1 = Metrics.macroF1(y_test, y_pred_test);

            // Count the SLOC
            int sloc = countSloc("models/LogisticRegression.java");

            // Display the results that I got
            System.out.println("\nOutputs:");
            System.out.println("*******");
            System.out.println("Algorithm: Logistic Regression");
            System.out.println("Train time: " + String.format("%.3f", trainTime) + " seconds");
            System.out.println("Metric 1: Accuracy: " + String.format("%.4f", accuracy));
            System.out.println("Metric 2: Macro-F1: " + String.format("%.4f", macroF1));
            System.out.println("Metric 3: SLOC: " + sloc);

            //Store result
            results.add(new AlgorithmResult("Java", "Logistic Regression", trainTime,
                                "Accuracy", accuracy, "Macro-F1", macroF1, sloc));

        } catch (Exception e) {
            System.err.println("Error during Logistic Regression: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void runKNN() {
        if (!dataLoaded) {
            System.out.println("Please load data first (option 1)");
            return;
        }
        System.out.println("\nk-Nearest Neighbors:");
        System.out.println("********************");
        System.out.print("Enter input options:\n");

        // Input option 1: the target variable
        System.out.println("Input option 1: Target variable: income");

        // Input option 2: the k value
        System.out.print("Input option 2: k = ");
        int k = 5;
        try {
            String input = scanner.nextLine().trim();
            k = Integer.parseInt(input);
            if (k <= 0) {
                System.out.println("  (Invalid k, using deafult k = 5)");
                k = 5;
            }
        } catch (NumberFormatException e) {
            System.out.println("5  --> number of neighbors (default)");
        }
        System.out.println("     --> number of neighbors");

        try {
            // Train the model
            long startTime = System.currentTimeMillis();

            KNearestNeighbors model = new KNearestNeighbors(k);
            model.fit(X_train, y_train);

            long trainEndTime = System.currentTimeMillis();

            // Make some kind of  predictions
            double[] y_pred_test = model.predict(X_test);

            long endTime = System.currentTimeMillis();
            double trainTime = (trainEndTime - startTime) / 1000.0;

            // Evaluate
            double accuracy = Metrics.accuracy(y_test, y_pred_test);
            double macroF1 = Metrics.macroF1(y_test, y_pred_test);

            //Count the SLOC
            int sloc = countSloc("models/KNearestNeighbors.java");

            // Display the results that I got
            System.out.println("\nOutputs:");
            System.out.println("*******");
            System.out.println("Algorithm: k-Nearest Neighbors");
            System.out.println("Train time: " + String.format("%.3f", trainTime) + " seconds");
            System.out.println("Metric 1: Accuracy: " + String.format("%.4f", accuracy));
            System.out.println("Metric 2: Macro-F1: " + String.format("%.4f", macroF1));
            System.out.println("Metric 3: SLOC: " + sloc);

            //Store result
            results.add(new AlgorithmResult("Java", "k-NN (k=" + k + ")", trainTime,
                                "Accuracy", accuracy, "Macro-F1", macroF1, sloc));

        } catch (Exception e) {
            System.err.println("Error during k-NN: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void runDecisionTree() {
        if (!dataLoaded) {
            System.out.println("Please load data first (option 1)");
            return;
        }

        System.out.println("\nDecision Tree (ID3):");
        System.out.println("********************");
        System.out.println("Enter input options:\n");
    
        System.out.println("Input option 1: Target variable: income");
    
        System.out.print("Input option 2: max_depth = ");
        int maxDepth = 10;
        try {
            maxDepth = Integer.parseInt(scanner.nextLine().trim());
            if (maxDepth <= 0) {
            System.out.println("  --> Invalid depth, using default max_depth = 10");
            maxDepth = 10;
        }
        } catch (NumberFormatException e) {
            System.out.println("10 (default)");
        }

         try {
        // Train the model (using one-hot encoded data which acts like categorical)
        long startTime = System.currentTimeMillis();

        DecisionTree model = new DecisionTree(maxDepth);
        model.fit(X_train, y_train);

        long endTime = System.currentTimeMillis();
        double trainTime = (endTime - startTime) / 1000.0;

        // Make predictions
        double[] y_pred_test = model.predict(X_test);

        // Evaluate
        double accuracy = Metrics.accuracy(y_test, y_pred_test);
        double macroF1 = Metrics.macroF1(y_test, y_pred_test);

        // Count SLOC
        int sloc = countSloc("models/DecisionTree.java");

        // Display results
        System.out.println("\nOutputs:");
        System.out.println("*******");
        System.out.println("Algorithm: Decision Tree (ID3)");
        System.out.println("Train time: " + String.format("%.3f", trainTime) + " seconds");
        System.out.println("Metric 1: Accuracy: " + String.format("%.4f", accuracy));
        System.out.println("Metric 2: Macro-F1: " + String.format("%.4f", macroF1));
        System.out.println("Metric 3: SLOC: " + sloc);

        //Store result
        results.add(new AlgorithmResult("Java", "Decision Tree (ID3)", trainTime,
                                "Accuracy", accuracy, "Macro-F1", macroF1, sloc));

    } catch (Exception e) {
        System.err.println("Error during Decision Tree: " + e.getMessage());
        e.printStackTrace();
    }
}
    
    private static void runNaiveBayes() {
        if (!dataLoaded) {
            System.out.println("Please load data first (option 1)");
            return;
        }
        
        System.out.println("\nGaussian Naive Bayes:");
        System.out.println("*********************");
        System.out.println("Enter input options:\n");
    
        System.out.println("Input option 1: Target variable: income");
        System.out.println("Input option 2: No additional parameters needed");
    
        try {
            Preprocessor nbPreprocessor = new Preprocessor(dataLoader);
            nbPreprocessor.preprocess("income", 0.8, 42, false);
        
            double[][] X_train_nb = nbPreprocessor.getXTrain();
            double[][] X_test_nb = nbPreprocessor.getXTest();
            double[] y_train_nb = nbPreprocessor.getYTrain();
            double[] y_test_nb = nbPreprocessor.getYTest();

            // Help train the model
            long startTime = System.currentTimeMillis();
        
            GaussianNaiveBayes model = new GaussianNaiveBayes();
            model.fit(X_train_nb, y_train_nb);
        
            long endTime = System.currentTimeMillis();
            double trainTime = (endTime - startTime) / 1000.0;
        
            // Make the predictions
            double[] y_pred_test = model.predict(X_test_nb);
        
            // Evaluate
            double accuracy = Metrics.accuracy(y_test_nb, y_pred_test);
            double macroF1 = Metrics.macroF1(y_test_nb, y_pred_test);
        
            // Count the SLOC
            int sloc = countSloc("models/GaussianNaiveBayes.java");
        
            // Display results that I got
            System.out.println("\nOutputs:");
            System.out.println("*******");
            System.out.println("Algorithm: Gaussian Naive Bayes");
            System.out.println("Train time: " + String.format("%.3f", trainTime) + " seconds");
            System.out.println("Metric 1: Accuracy: " + String.format("%.4f", accuracy));
            System.out.println("Metric 2: Macro-F1: " + String.format("%.4f", macroF1));
            System.out.println("Metric 3: SLOC: " + sloc);

            //Store result
            results.add(new AlgorithmResult("Java", "Gaussian Naive Bayes", trainTime,
                                "Accuracy", accuracy, "Macro-F1", macroF1, sloc));
        
        } catch (Exception e) {
            System.err.println("Error during Gaussian Naive Bayes: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    
    //For the SLOC
    private static int countSloc(String filepath) {
    int sloc = 0;
    boolean inMultilineComment = false;

    try (java.io.BufferedReader reader = new java.io.BufferedReader(
            new java.io.FileReader(filepath))) {
        String line;

        while ((line = reader.readLine()) != null) {
            String trimmed = line.trim();

            // Check for multiline comments
            if (trimmed.startsWith("/*")) {
                inMultilineComment = true;
            }

            if (inMultilineComment) {
                if (trimmed.endsWith("*/")) {
                    inMultilineComment = false;
                }
                continue;
            }

            // Skip lines that are blans
            if (trimmed.isEmpty() || trimmed.startsWith("//") ||
                trimmed.equals("*") || trimmed.startsWith("* ")) {
                continue;
            }

            sloc++;
        }

    } catch (Exception e) {
        System.err.println("Error counting SLOC: " + e.getMessage());
        return 0;
    }

    return sloc;
}

    private static void printResults() {
    String implName = results.isEmpty() ? "<Implementation Name>" : results.get(0).getImplementation();

    System.out.println("\n" + implName + " Results:");
    System.out.println("  ******************************");

    if (results.isEmpty()) {
        System.out.println("No results yet. Run some algorithms first!");
        return;
    }

    // Print the header
    System.out.println(String.format(
        "  %-24s %-22s %-16s %-16s %-16s %-10s",
        "Impl", "Algorithm", "TrainTime", "TestMetric1", "TestMetric2", "SLOC"
    ));
    

    // Print the result result
    for (AlgorithmResult result : results) {
        System.out.println(String.format(
            "  %-24s %-22s %-16.3f %-16.4f %-16.4f %-10d",
            result.getImplementation(),
            result.getAlgorithm(),
            result.getTrainTime(),
            result.getMetric1(),
            result.getMetric2(),
            result.getSloc()
        ));
    }
  }
}
