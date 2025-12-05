#include "LogisticRegression.h"
#include <cmath>
#include <numeric>
#include <algorithm>

double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

LogisticModel fit_logistic(const std::vector<std::vector<double>>& X,
                           const std::vector<int>& y,
                           double lr, int epochs, double reg) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    
    LogisticModel model;
    model.weights.assign(n_features, 0.0);
    model.bias = 0.0;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < n_samples; ++i) {
            double z = dot(model.weights, X[i]) + model.bias;
            double pred = sigmoid(z);
            double error = pred - y[i];
            
            for (size_t j = 0; j < n_features; ++j)
                model.weights[j] -= lr * (error * X[i][j] + reg * model.weights[j]);
            
            model.bias -= lr * error;
        }
    }
    
    return model;
}

double predict_proba(const LogisticModel& model, const std::vector<double>& x) {
    return sigmoid(dot(model.weights, x) + model.bias);
}

std::vector<int> predict_logistic(const LogisticModel& model,
                                  const std::vector<std::vector<double>>& X) {
    std::vector<int> y_pred;
    for (const auto& row : X)
        y_pred.push_back(predict_proba(model, row) >= 0.5 ? 1 : 0);
    return y_pred;
}

double computeAccuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i])
            ++correct;
    return double(correct) / y_true.size();
}

double macroF1(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
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
