#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include <string>
#include <memory>
#include <map>

struct TreeNode {
    bool isLeaf;
    int label;
    int featureIndex;
    double threshold;
    bool isNumeric;
    std::map<std::string, std::shared_ptr<TreeNode>> children;
    std::shared_ptr<TreeNode> left;
    std::shared_ptr<TreeNode> right;
    
    TreeNode() : isLeaf(false), label(-1), featureIndex(-1), 
                 threshold(0.0), isNumeric(false) {}
};

struct DecisionTreeModel {
    std::shared_ptr<TreeNode> root;
};

DecisionTreeModel fit_tree(const std::vector<std::vector<double>>& X, 
                           const std::vector<int>& y, int maxDepth = 10);

std::vector<int> predict_tree(const DecisionTreeModel& model, 
                              const std::vector<std::vector<double>>& X);

double computeAccuracy_tree(const std::vector<int>& y_true, 
                            const std::vector<int>& y_pred);

double macroF1_tree(const std::vector<int>& y_true, 
                    const std::vector<int>& y_pred);

#endif
