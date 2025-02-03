#ifndef KDE_NB_H
#define KDE_NB_H

#include "data_loader.h"

typedef struct {
    double ***samples;       // samples[class][feature][sample_idx]
    int class_counts[2];     // Number of samples per class
    double **bandwidth;      // bandwidth[class][feature]
    double prior[2];         // Prior probabilities
    int n_features;
} KDEModel;

void train_kde_nb(const Dataset *train_data, KDEModel *model);
int predict_kde_nb(const KDEModel *model, const double *sample);
void free_kde_model(KDEModel *model);

#endif // KDE_NB_H