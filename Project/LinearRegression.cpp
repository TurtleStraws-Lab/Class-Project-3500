#include "LinearRegression.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& X) {
    if (X.empty()) return {};
    size_t rows = X.size();
    size_t cols = X[0].size();

    std::vector<std::vector<double>> T(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            T[j][i] = X[i][j];

    return T;
}

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A,
                                        const std::vector<std::vector<double>>& B) {
    if (A.empty() || B.empty()) return {};

    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();

    std::vector<std::vector<double>> C(n, std::vector<double>(p, 0.0));

    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < m; k++)
            for (size_t j = 0; j < p; j++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

std::vector<std::vector<double>> addLambda(const std::vector<std::vector<double>>& XTX,
                                           double lambda) {
    std::vector<std::vector<double>> A = XTX;
    for (size_t i = 0; i < A.size(); i++)
        A[i][i] += lambda;
    return A;
}

std::vector<double> solveLinearSystem(std::vector<std::vector<double>> A,
                                      std::vector<double> b) {
    size_t n = A.size();

    for (size_t i = 0; i < n; i++) {
        double pivot = A[i][i];
        if (std::abs(pivot) < 1e-12) 
            throw std::runtime_error("Singular matrix");

        for (size_t j = i; j < n; j++) A[i][j] /= pivot;
        b[i] /= pivot;

        for (size_t k = i + 1; k < n; k++) {
            double factor = A[k][i];
            for (size_t j = i; j < n; j++)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }

    std::vector<double> x(n);
    for (int i = (int)n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (size_t j = i + 1; j < n; j++)
            x[i] -= A[i][j] * x[j];
    }

    return x;
}

LinearModel fit_linear(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y,
                       double lambda) {
    LinearModel model;

    size_t n = X.size();
    size_t d = X[0].size();

    std::vector<std::vector<double>> Xb(n, std::vector<double>(d + 1, 1.0));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < d; j++)
            Xb[i][j] = X[i][j];

    auto X_T = transpose(Xb);
    auto XTX = matmul(X_T, Xb);

    std::vector<double> XTy_vec(d + 1, 0.0);
    for (size_t i = 0; i < d + 1; i++)
        for (size_t j = 0; j < n; j++)
            XTy_vec[i] += X_T[i][j] * y[j];

    auto A = XTX;
    for (size_t i = 1; i < A.size(); i++)
        A[i][i] += lambda;

    model.weights = solveLinearSystem(A, XTy_vec);
    return model;
}

std::vector<double> predict_linear(const LinearModel& model,
                                   const std::vector<std::vector<double>>& X) {
    size_t n = X.size();
    size_t d = X[0].size();

    std::vector<double> preds(n, 0.0);

    for (size_t i = 0; i < n; i++) {
        double y = model.weights[d];
        for (size_t j = 0; j < d; j++)
            y += model.weights[j] * X[i][j];
        preds[i] = y;
    }

    return preds;
}

double computeRMSE(const std::vector<double>& y_true,
                   const std::vector<double>& y_pred) {
    size_t n = y_true.size();
    double sum = 0.0;

    for (size_t i = 0; i < n; i++)
        sum += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);

    return std::sqrt(sum / n);
}
