#include <math.h>
#include <stdlib.h>
#include "gnb.h"
#include "utils.h"

void train_gnb(const Dataset *train_data, GNBModel *model) {
    model->n_features = train_data->n_features;

    // Count samples per class
    int class_counts[2] = {0};
    for (int i = 0; i < train_data->n_samples; i++) {
        class_counts[train_data->labels[i]]++;
    }

    // Compute priors
    model->prior[0] = (double)class_counts[0] / train_data->n_samples;
    model->prior[1] = (double)class_counts[1] / train_data->n_samples;

    // Initialize mean and variance
    for (int c = 0; c < 2; c++) {
        for (int f = 0; f < model->n_features; f++) {
            model->mean[c][f] = 0.0;
            model->variance[c][f] = 0.0;
        }
    }

    // Compute mean
    for (int i = 0; i < train_data->n_samples; i++) {
        int label = train_data->labels[i];
        for (int f = 0; f < model->n_features; f++) {
            model->mean[label][f] += train_data->features[i][f];
        }
    }

    for (int c = 0; c < 2; c++) {
        if (class_counts[c] > 0) {
            for (int f = 0; f < model->n_features; f++) {
                model->mean[c][f] /= class_counts[c];
            }
        }
    }

    // Compute variance
    for (int i = 0; i < train_data->n_samples; i++) {
        int label = train_data->labels[i];
        for (int f = 0; f < model->n_features; f++) {
            double diff = train_data->features[i][f] - model->mean[label][f];
            model->variance[label][f] += diff * diff;
        }
    }

    for (int c = 0; c < 2; c++) {
        if (class_counts[c] > 1) {
            for (int f = 0; f < model->n_features; f++) {
                model->variance[c][f] = (model->variance[c][f] / (class_counts[c] - 1)) + 1e-9; // Add epsilon
            }
        }
    }
}

int predict_gnb(const GNBModel *model, const double *sample) {
    double log_probs[2] = {log(model->prior[0]), log(model->prior[1])};

    for (int c = 0; c < 2; c++) {
        for (int f = 0; f < model->n_features; f++) {
            double x = sample[f];
            double mean = model->mean[c][f];
            double var = model->variance[c][f];
            log_probs[c] += log(gaussian_pdf(x, mean, var));
        }
    }

    return (log_probs[1] > log_probs[0]) ? 1 : 0;
}
