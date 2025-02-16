#ifndef SNB_H
#define SNB_H

#include "data_loader.h"

typedef struct {
    double **means;       // Mean values for each class and feature
    double **variances;   // Variance for each class and feature
    double *class_priors; // Prior probabilities for each class
    int n_classes;        // Number of classes
    int n_features;       // Number of features
} SNBModel;

void train_snb(Dataset *data, SNBModel *model);
double** get_snb_probs(SNBModel *model, Dataset *data);
void free_snb_model(SNBModel *model);

#endif