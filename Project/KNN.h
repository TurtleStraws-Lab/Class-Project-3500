#ifndef KNN_H
#define KNN_H

#include <vector>

struct KNNModel {
    int k;
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
};

KNNModel fit_knn(const std::vector<std::vector<double>>& X_train, 
                 const std::vector<int>& y_train, int k);

std::vector<int> predict_knn(const KNNModel& model, 
                              const std::vector<std::vector<double>>& X_test);

double macroF1_knn(const std::vector<int>& y_true,
                   const std::vector<int>& y_pred);

#endif
