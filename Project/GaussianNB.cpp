#include "GaussianNB.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <map>
#include <cmath>

double gaussian_prob(double x, double mean, double var) {
    if (var == 0) var = 1e-6;
    double exponent = std::exp(-std::pow(x - mean, 2) / (2 * var));
    return (1.0 / std::sqrt(2 * M_PI * var)) * exponent;
}

GaussianNBModel fit_gnb(const std::vector<std::vector<double>>& X,
                        const std::vector<int>& y) {
    GaussianNBModel model;
    std::map<int, std::vector<std::vector<double>>> class_samples;
    int n_features = X[0].size();
    
    for (size_t i = 0; i < y.size(); ++i)
        class_samples[y[i]].push_back(X[i]);
    
    model.classes.reserve(class_samples.size());
    model.means.reserve(class_samples.size());
    model.variances.reserve(class_samples.size());
    model.priors.reserve(class_samples.size());
    
    for (auto& [cls, samples] : class_samples) {
        model.classes.push_back(cls);
        int n_samples = samples.size();
        
        std::vector<double> mean(n_features, 0.0);
        std::vector<double> var(n_features, 0.0);
        
        for (int j = 0; j < n_features; ++j)
            for (auto& sample : samples)
                mean[j] += sample[j];
        
        for (int j = 0; j < n_features; ++j)
            mean[j] /= n_samples;
        
        for (int j = 0; j < n_features; ++j)
            for (auto& sample : samples)
                var[j] += (sample[j] - mean[j]) * (sample[j] - mean[j]);
        
        for (int j = 0; j < n_features; ++j)
            var[j] /= n_samples;
        
        model.means.push_back(mean);
        model.variances.push_back(var);
        model.priors.push_back(static_cast<double>(n_samples) / y.size());
    }
    
    return model;
}

std::vector<int> predict_gnb(const GaussianNBModel& model,
                             const std::vector<std::vector<double>>& X) {
    std::vector<int> y_pred;
    
    for (auto& row : X) {
        double best_prob = -1.0;
        int best_class = model.classes[0];
        
        for (size_t i = 0; i < model.classes.size(); ++i) {
            double prob = std::log(model.priors[i]);
            for (size_t j = 0; j < row.size(); ++j)
                prob += std::log(gaussian_prob(row[j], model.means[i][j], 
                                              model.variances[i][j]));
            
            if (i == 0 || prob > best_prob) {
                best_prob = prob;
                best_class = model.classes[i];
            }
        }
        y_pred.push_back(best_class);
    }
    
    return y_pred;
}

double macroF1_gnb(const std::vector<int>& y_true,
                   const std::vector<int>& y_pred) {
    std::map<int, int> tp, fp, fn;
    std::vector<int> classes;
    
    for (auto& y : y_true)
        if (std::find(classes.begin(), classes.end(), y) == classes.end())
            classes.push_back(y);
    
    for (auto cls : classes) {
        tp[cls] = 0;
        fp[cls] = 0;
        fn[cls] = 0;
    }
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) 
            tp[y_true[i]]++;
        else {
            fp[y_pred[i]]++;
            fn[y_true[i]]++;
        }
    }
    
    double f1_sum = 0.0;
    for (auto cls : classes) {
        double precision = tp[cls] + fp[cls] == 0 ? 0 : (double)tp[cls] / (tp[cls] + fp[cls]);
        double recall = tp[cls] + fn[cls] == 0 ? 0 : (double)tp[cls] / (tp[cls] + fn[cls]);
        double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);
        f1_sum += f1;
    }
    
    return f1_sum / classes.size();
}
