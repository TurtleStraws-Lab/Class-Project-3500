#ifndef LOADDATA_H
#define LOADDATA_H

#include <string>
#include <vector>

// Dataset struct
struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::string> headers;
    bool loaded = false;

    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> X_test;
    std::vector<int> y_train;
    std::vector<int> y_test;
};

extern Dataset dataset;

// Functions
void loadData(const std::string& filename);
void splitDataset(double trainFraction = 0.8);

#endif
