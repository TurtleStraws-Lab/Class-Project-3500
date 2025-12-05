// Procedural.cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "loadData.h"
#include "LinearRegression.h"
#include "LogisticRegression.h"
#include "KNN.h"
#include "DecisionTree.h"
#include "GaussianNB.h"

enum AlgorithmType { NONE, LINEAR, LOGISTIC, KNN_ALGO, TREE, NB };
AlgorithmType lastTrainedAlgo = NONE;
double lastTrainTime = 0.0;

LinearModel linear_model;
LogisticModel logistic_model;
KNNModel knn_model;
DecisionTreeModel tree_model;
GaussianNBModel gnb_model;

static std::vector<double> ints_to_doubles(const std::vector<int>& v) {
    return std::vector<double>(v.begin(), v.end());
}

void printResults() {
    std::cout << "\n================= Results =================\n";
    switch (lastTrainedAlgo) {
        case LINEAR: {
            std::vector<double> y_test_d = ints_to_doubles(dataset.y_test);
            std::vector<double> y_pred = predict_linear(linear_model, dataset.X_test);
            double rmse = computeRMSE(y_test_d, y_pred);
            std::cout << "Algorithm: Linear Regression\n";
            std::cout << "Training Time: " << std::fixed << std::setprecision(6) << lastTrainTime << " seconds\n";
            std::cout << "Test RMSE: " << rmse << "\n";
            break;
        }
        case LOGISTIC: {
            std::vector<int> y_pred = predict_logistic(logistic_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1(dataset.y_test, y_pred);
            std::cout << "Algorithm: Logistic Regression\n";
            std::cout << "Training Time: " << std::fixed << std::setprecision(6) << lastTrainTime << " seconds\n";
            std::cout << "Test Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1 Score: " << f1 << "\n";
            break;
        }
        case KNN_ALGO: {
            std::vector<int> y_pred = predict_knn(knn_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1_knn(dataset.y_test, y_pred);
            std::cout << "Algorithm: k-Nearest Neighbors\n";
            std::cout << "Training Time: " << std::fixed << std::setprecision(6) << lastTrainTime << " seconds\n";
            std::cout << "Test Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1 Score: " << f1 << "\n";
            break;
        }
        case TREE: {
            std::vector<int> y_pred = predict_tree(tree_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1_tree(dataset.y_test, y_pred);
            std::cout << "Algorithm: Decision Tree (ID3)\n";
            std::cout << "Training Time: " << std::fixed << std::setprecision(6) << lastTrainTime << " seconds\n";
            std::cout << "Test Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1 Score: " << f1 << "\n";
            break;
        }
        case NB: {
            std::vector<int> y_pred = predict_gnb(gnb_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1_gnb(dataset.y_test, y_pred);
            std::cout << "Algorithm: Gaussian Naive Bayes\n";
            std::cout << "Training Time: " << std::fixed << std::setprecision(6) << lastTrainTime << " seconds\n";
            std::cout << "Test Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1 Score: " << f1 << "\n";
            break;
        }
        default:
            std::cout << "No algorithm trained yet.\n";
            break;
    }
    std::cout << "===========================================\n\n";
}

int main() {
    while (true) {
        std::cout << "======================================\n";
        std::cout << "  C++ Procedural ML Project\n";
        std::cout << "======================================\n";
        std::cout << "(1) Load data\n";
        std::cout << "(2) Linear Regression (closed-form)\n";
        std::cout << "(3) Logistic Regression (binary)\n";
        std::cout << "(4) k-Nearest Neighbors\n";
        std::cout << "(5) Decision Tree (ID3)\n";
        std::cout << "(6) Gaussian Naive Bayes\n";
        std::cout << "(7) Print results\n";
        std::cout << "(8) Quit\n";
        std::cout << "Enter choice: ";

        int choice;
        if (!(std::cin >> choice)) { 
            std::cin.clear(); 
            std::cin.ignore(10000, '\n'); 
            continue; 
        }

        if (choice == 1) {
            std::string filename;
            std::cout << "Enter CSV filename: ";
            std::cin >> filename;
            loadData(filename);
            splitDataset();
            std::cout << "Loaded " << dataset.X.size() << " samples.\n";
            if (!dataset.X.empty()) 
                std::cout << "Feature count: " << dataset.X[0].size() << "\n";
        }
        else if (choice == 2) {
            if (dataset.X_train.empty()) { 
                std::cout << "Load data first.\n"; 
                continue; 
            }
            
            std::vector<double> y_train_d = ints_to_doubles(dataset.y_train);
            std::cout << "Training Linear Regression (closed-form, L2 lambda=0.1)...\n";
            
            auto t0 = std::chrono::high_resolution_clock::now();
            linear_model = fit_linear(dataset.X_train, y_train_d, 0.1);
            auto t1 = std::chrono::high_resolution_clock::now();
            lastTrainTime = std::chrono::duration<double>(t1 - t0).count();
            lastTrainedAlgo = LINEAR;
            
            std::vector<double> y_test_d = ints_to_doubles(dataset.y_test);
            std::vector<double> y_pred = predict_linear(linear_model, dataset.X_test);
            double rmse = computeRMSE(y_test_d, y_pred);
            std::cout << "Linear Regression RMSE: " << rmse << "\n";
            std::cout << "Training time: " << lastTrainTime << " seconds\n";
        }
        else if (choice == 3) {
            if (dataset.X_train.empty()) { 
                std::cout << "Load data first.\n"; 
                continue; 
            }
            
            std::cout << "Training Logistic Regression (GD, L2 optional)...\n";
            auto t0 = std::chrono::high_resolution_clock::now();
            logistic_model = fit_logistic(dataset.X_train, dataset.y_train, 0.01, 100, 0.0);
            auto t1 = std::chrono::high_resolution_clock::now();
            lastTrainTime = std::chrono::duration<double>(t1 - t0).count();
            lastTrainedAlgo = LOGISTIC;
            
            std::vector<int> y_pred = predict_logistic(logistic_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1(dataset.y_test, y_pred);
            std::cout << "Logistic Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1: " << f1 << "\n";
            std::cout << "Training time: " << lastTrainTime << " seconds\n";
        }
        else if (choice == 4) {
            if (dataset.X_train.empty()) { 
                std::cout << "Load data first.\n"; 
                continue; 
            }
            
            int k = 5;
            std::cout << "Training k-NN (k=" << k << ")...\n";
            auto t0 = std::chrono::high_resolution_clock::now();
            knn_model = fit_knn(dataset.X_train, dataset.y_train, k);
            auto t1 = std::chrono::high_resolution_clock::now();
            lastTrainTime = std::chrono::duration<double>(t1 - t0).count();
            lastTrainedAlgo = KNN_ALGO;
            
            std::vector<int> y_pred = predict_knn(knn_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1_knn(dataset.y_test, y_pred);
            std::cout << "KNN Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1: " << f1 << "\n";
            std::cout << "Training time: " << lastTrainTime << " seconds\n";
        }
        else if (choice == 5) {
            if (dataset.X_train.empty()) { 
                std::cout << "Load data first.\n"; 
                continue; 
            }
            
            std::cout << "Training Decision Tree (ID3)...\n";
            auto t0 = std::chrono::high_resolution_clock::now();
            tree_model = fit_tree(dataset.X_train, dataset.y_train);
            auto t1 = std::chrono::high_resolution_clock::now();
            lastTrainTime = std::chrono::duration<double>(t1 - t0).count();
            lastTrainedAlgo = TREE;
            
            std::vector<int> y_pred = predict_tree(tree_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1_tree(dataset.y_test, y_pred);
            std::cout << "Decision Tree Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1: " << f1 << "\n";
            std::cout << "Training time: " << lastTrainTime << " seconds\n";
        }
        else if (choice == 6) {
            if (dataset.X_train.empty()) { 
                std::cout << "Load data first.\n"; 
                continue; 
            }
            
            std::cout << "Training Gaussian Naive Bayes...\n";
            auto t0 = std::chrono::high_resolution_clock::now();
            gnb_model = fit_gnb(dataset.X_train, dataset.y_train);
            auto t1 = std::chrono::high_resolution_clock::now();
            lastTrainTime = std::chrono::duration<double>(t1 - t0).count();
            lastTrainedAlgo = NB;
            
            std::vector<int> y_pred = predict_gnb(gnb_model, dataset.X_test);
            double acc = computeAccuracy(dataset.y_test, y_pred);
            double f1 = macroF1_gnb(dataset.y_test, y_pred);
            std::cout << "GNB Accuracy: " << acc * 100.0 << "%\n";
            std::cout << "Macro-F1: " << f1 << "\n";
            std::cout << "Training time: " << lastTrainTime << " seconds\n";
        }
        else if (choice == 7) {
            printResults();
        }
        else if (choice == 8) {
            std::cout << "Quitting.\n";
            break;
        }
        else {
            std::cout << "Invalid option.\n";
        }
    }

    return 0;
}
