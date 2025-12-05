#include "loadData.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <random>

Dataset dataset;

void loadData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    
    dataset.X.clear();
    dataset.y.clear();
    dataset.headers.clear();
    
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
            dataset.headers.push_back(token);
        }
    }
    
    std::cout << "Columns found:\n";
    for (size_t i = 0; i < dataset.headers.size(); i++)
        std::cout << i << ": " << dataset.headers[i] << "\n";
    
    int targetCol;
    std::cout << "Enter the column number to use as target: ";
    std::cin >> targetCol;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;
        int colIndex = 0;
        
        while (std::getline(ss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
            if (colIndex == targetCol) {
                dataset.y.push_back(token == ">50K" ? 1 : 0);
            } else {
                try { 
                    row.push_back(std::stod(token)); 
                }
                catch (...) { 
                    row.push_back(0.0); 
                }
            }
            colIndex++;
        }
        dataset.X.push_back(row);
    }
    
    dataset.loaded = true;
    std::cout << "Loaded " << dataset.X.size() << " samples with " 
              << dataset.X[0].size() << " features.\n";
}

void splitDataset(double trainFraction) {
    if (!dataset.loaded) return;
    
    size_t n = dataset.X.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    size_t trainSize = static_cast<size_t>(n * trainFraction);
    
    dataset.X_train.clear();
    dataset.y_train.clear();
    dataset.X_test.clear();
    dataset.y_test.clear();
    
    for (size_t i = 0; i < n; ++i) {
        if (i < trainSize) {
            dataset.X_train.push_back(dataset.X[indices[i]]);
            dataset.y_train.push_back(dataset.y[indices[i]]);
        } else {
            dataset.X_test.push_back(dataset.X[indices[i]]);
            dataset.y_test.push_back(dataset.y[indices[i]]);
        }
    }
}
