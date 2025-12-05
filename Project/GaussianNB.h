#ifndef GAUSSIANNB_H
#define GAUSSIANNB_H

#include <vector>
#include <cmath>
#include <map>

struct GaussianNBModel {
    std::vector<int> classes;
    std::vector<std::vector<double>> means;
    std::vector<std::vector<double>> variances;
    std::vector<double> priors;
};

GaussianNBModel fit_gnb(const std::vector<std::vector<double>>& X,
                        const std::vector<int>& y);

std::vector<int> predict_gnb(const GaussianNBModel& model,
                             const std::vector<std::vector<double>>& X);

double macroF1_gnb(const std::vector<int>& y_true,
                   const std::vector<int>& y_pred);

#endif
