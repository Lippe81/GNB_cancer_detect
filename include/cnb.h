#ifndef CNB_H
#define CNB_H

#include "data_loader.h"

typedef struct {
    double **feature_log_prob;  // Log probabilities for features
    double *class_log_prior;    // Log prior probabilities
    int n_classes;
    int n_features;
} CNBModel;

void train_cnb(Dataset *data, CNBModel *model);
double** get_cnb_probs(CNBModel *model, Dataset *data);
void free_cnb_model(CNBModel *model);

#endif