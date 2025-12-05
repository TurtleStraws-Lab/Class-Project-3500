#include "DecisionTree.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <limits>
#include <iostream>

double entropy(const std::vector<int>& y) {
    std::map<int,int> counts;
    for (int label : y) counts[label]++;
    
    double H = 0.0;
    int n = y.size();
    for (auto& p : counts) {
        double prob = double(p.second)/n;
        H -= prob * std::log2(prob);
    }
    return H;
}

double infoGain(const std::vector<int>& y, const std::vector<int>& left, const std::vector<int>& right) {
    double H = entropy(y);
    double H_left = entropy(left);
    double H_right = entropy(right);
    return H - (left.size()/double(y.size()) * H_left + right.size()/double(y.size()) * H_right);
}

std::shared_ptr<TreeNode> buildTree(const std::vector<std::vector<double>>& X, 
                                    const std::vector<int>& y, 
                                    int depth, int maxDepth) {
    auto node = std::make_shared<TreeNode>();
    
    // Check stopping conditions
    std::set<int> unique_labels(y.begin(), y.end());
    if (unique_labels.size() == 1 || depth >= maxDepth) {
        node->isLeaf = true;
        node->label = y[0];
        return node;
    }

    // Find best split
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -1.0;

    int numFeatures = X[0].size();
    for (int f = 0; f < numFeatures; ++f) {
        std::vector<double> values;
        for (auto& row : X) values.push_back(row[f]);
        std::set<double> unique(values.begin(), values.end());

        for (double threshold : unique) {
            std::vector<int> left_labels, right_labels;
            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][f] <= threshold) 
                    left_labels.push_back(y[i]);
                else 
                    right_labels.push_back(y[i]);
            }
            
            if (left_labels.empty() || right_labels.empty()) continue;
            
            double gain = infoGain(y, left_labels, right_labels);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = f;
                bestThreshold = threshold;
            }
        }
    }

    // If no good split found, make it a leaf
    if (bestFeature == -1) {
        node->isLeaf = true;
        node->label = y[0];
        return node;
    }

    // Store split info
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->isNumeric = true;

    // Actually split the data
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<int> y_left, y_right;
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][bestFeature] <= bestThreshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    // Recursively build subtrees
    node->left = buildTree(X_left, y_left, depth+1, maxDepth);
    node->right = buildTree(X_right, y_right, depth+1, maxDepth);

    return node;
}

DecisionTreeModel fit_tree(const std::vector<std::vector<double>>& X, 
                           const std::vector<int>& y, int maxDepth) {
    DecisionTreeModel model;
    model.root = buildTree(X, y, 0, maxDepth);
    return model;
}

int predict_node(const std::shared_ptr<TreeNode>& node, const std::vector<double>& x) {
    if (node->isLeaf) return node->label;
    
    if (x[node->featureIndex] <= node->threshold)
        return predict_node(node->left, x);
    else
        return predict_node(node->right, x);
}

std::vector<int> predict_tree(const DecisionTreeModel& model, const std::vector<std::vector<double>>& X) {
    std::vector<int> y_pred;
    for (auto& row : X) 
        y_pred.push_back(predict_node(model.root, row));
    return y_pred;
}

double computeAccuracy_tree(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i]) correct++;
    return double(correct)/y_true.size() * 100.0;
}

double macroF1_tree(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    std::set<int> labels(y_true.begin(), y_true.end());
    double f1sum = 0.0;
    
    for (int label : labels) {
        int tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_pred[i] == label && y_true[i] == label) tp++;
            if (y_pred[i] == label && y_true[i] != label) fp++;
            if (y_pred[i] != label && y_true[i] == label) fn++;
        }
        
        double prec = tp + fp == 0 ? 0.0 : double(tp)/(tp+fp);
        double rec = tp + fn == 0 ? 0.0 : double(tp)/(tp+fn);
        double f1 = (prec+rec==0) ? 0.0 : 2*prec*rec/(prec+rec);
        f1sum += f1;
    }
    
    return f1sum/labels.size();
}
