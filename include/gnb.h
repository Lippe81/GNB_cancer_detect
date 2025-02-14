#ifndef GNB_H
#define GNB_H

#include "data_loader.h"

typedef struct {
    double mean[2][30];      // Mean for class 0 (B) and 1 (M) for each feature
    double variance[2][30];  // Variance for each class and feature
    double prior[2];         // Prior probabilities P(B) and P(M)
    int n_features;
} GNBModel;

void train_gnb(const Dataset *train_data, GNBModel *model);
int predict_gnb(const GNBModel *model, const double *sample);
double **get_gnb_probs(const GNBModel *model, const Dataset *data); // Add this line

#endif // GNB_H