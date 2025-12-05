#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>

struct LogisticModel {
    std::vector<double> weights;
    double bias;
};

LogisticModel fit_logistic(const std::vector<std::vector<double>>& X,
                           const std::vector<int>& y,
                           double lr, int epochs, double reg);

std::vector<int> predict_logistic(const LogisticModel& model,
                                  const std::vector<std::vector<double>>& X);

double predict_proba(const LogisticModel& model, const std::vector<double>& x);

double computeAccuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred);

double macroF1(const std::vector<int>& y_true, const std::vector<int>& y_pred);

double sigmoid(double z);

double dot(const std::vector<double>& a, const std::vector<double>& b);

#endif
