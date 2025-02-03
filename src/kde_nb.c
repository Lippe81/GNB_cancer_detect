#include <math.h>
#include <stdlib.h>
#include "kde_nb.h"
#include "utils.h"

// Compute standard deviation of an array
static double compute_std(const double *samples, int n) {
    if (n <= 1) return 1e-9; // Avoid division by zero
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += samples[i];
    }
    mean /= n;

    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        variance += pow(samples[i] - mean, 2);
    }
    variance /= (n - 1); // Sample variance
    return sqrt(variance) + 1e-9;
}

void train_kde_nb(const Dataset *train_data, KDEModel *model) {
    model->n_features = train_data->n_features;

    // Count samples per class
    model->class_counts[0] = 0;
    model->class_counts[1] = 0;
    for (int i = 0; i < train_data->n_samples; i++) {
        model->class_counts[train_data->labels[i]]++;
    }

    // Compute priors
    model->prior[0] = (double)model->class_counts[0] / train_data->n_samples;
    model->prior[1] = (double)model->class_counts[1] / train_data->n_samples;

    // Allocate memory for samples
    model->samples = malloc(2 * sizeof(double **));
    for (int c = 0; c < 2; c++) {
        model->samples[c] = malloc(model->n_features * sizeof(double *));
        for (int f = 0; f < model->n_features; f++) {
            model->samples[c][f] = malloc(model->class_counts[c] * sizeof(double));
        }
    }

    // Populate samples
    int idx[2] = {0}; // Index trackers for each class
    for (int i = 0; i < train_data->n_samples; i++) {
        int label = train_data->labels[i];
        for (int f = 0; f < model->n_features; f++) {
            model->samples[label][f][idx[label]] = train_data->features[i][f];
        }
        idx[label]++;
    }

    // Allocate bandwidth and compute using Silverman's rule
    model->bandwidth = malloc(2 * sizeof(double *));
    for (int c = 0; c < 2; c++) {
        model->bandwidth[c] = malloc(model->n_features * sizeof(double));
        for (int f = 0; f < model->n_features; f++) {
            double std_dev = compute_std(model->samples[c][f], model->class_counts[c]);
            double n = model->class_counts[c];
            model->bandwidth[c][f] = 1.06 * std_dev * pow(n, -0.2);
        }
    }
}

int predict_kde_nb(const KDEModel *model, const double *sample) {
    double log_probs[2] = {log(model->prior[0]), log(model->prior[1])};

    for (int c = 0; c < 2; c++) {
        for (int f = 0; f < model->n_features; f++) {
            int n = model->class_counts[c];
            double h = model->bandwidth[c][f];
            double sum = 0.0;

            for (int i = 0; i < n; i++) {
                double diff = sample[f] - model->samples[c][f][i];
                double u = diff / h;
                sum += exp(-0.5 * u * u);
            }

            // Compute KDE and avoid log(0)
            double kde = sum / (n * h * sqrt(2 * M_PI));
            log_probs[c] += log(kde + 1e-9);
        }
    }

    return (log_probs[1] > log_probs[0]) ? 1 : 0;
}

void free_kde_model(KDEModel *model) {
    for (int c = 0; c < 2; c++) {
        for (int f = 0; f < model->n_features; f++) {
            free(model->samples[c][f]);
        }
        free(model->samples[c]);
        free(model->bandwidth[c]);
    }
    free(model->samples);
    free(model->bandwidth);
}