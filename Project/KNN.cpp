#include "KNN.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>

KNNModel fit_knn(const std::vector<std::vector<double>>& X,
                 const std::vector<int>& y,
                 int k) {
    KNNModel model;
    model.X_train = X;
    model.y_train = y;
    model.k = k;
    return model;
}

double euclidean(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

std::vector<int> predict_knn(const KNNModel& model,
                             const std::vector<std::vector<double>>& X_test) {
    std::vector<int> y_pred;
    
    for (const auto& x : X_test) {
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < model.X_train.size(); ++i)
            distances.emplace_back(euclidean(x, model.X_train[i]), model.y_train[i]);
        
        std::sort(distances.begin(), distances.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        std::map<int, int> counts;
        for (int i = 0; i < model.k && i < (int)distances.size(); ++i)
            counts[distances[i].second]++;
        
        int max_count = -1, pred = -1;
        for (auto& p : counts) {
            if (p.second > max_count) {
                max_count = p.second;
                pred = p.first;
            }
        }
        
        y_pred.push_back(pred);
    }
    
    return y_pred;
}

double macroF1_knn(const std::vector<int>& y_true,
                   const std::vector<int>& y_pred) {
    int tp0 = 0, fp0 = 0, fn0 = 0;
    int tp1 = 0, fp1 = 0, fn1 = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 0) {
            if (y_pred[i] == 0) tp0++; 
            else fn0++;
        } else {
            if (y_pred[i] == 1) tp1++; 
            else fn1++;
        }
        
        if (y_true[i] == 0 && y_pred[i] == 1) fp0++;
        if (y_true[i] == 1 && y_pred[i] == 0) fp1++;
    }
    
    double f1_0 = tp0 == 0 ? 0.0 : 2.0*tp0 / (2*tp0 + fp0 + fn0);
    double f1_1 = tp1 == 0 ? 0.0 : 2.0*tp1 / (2*tp1 + fp1 + fn1);
    
    return (f1_0 + f1_1) / 2.0;
}
