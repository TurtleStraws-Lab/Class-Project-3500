#ifndef LOADDATA_H
#define LOADDATA_H

#include <vector>
#include <string>

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::string> headers;
    
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<int> y_test;
    
    bool loaded = false;
};

extern Dataset dataset;

void loadData(const std::string& filename);

void splitDataset(double trainFraction = 0.8);

#endif
