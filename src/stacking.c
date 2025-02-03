#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "stacking.h"
#include "utils.h"

// Sigmoid function for logistic regression
static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

StackedFeatures init_stacked_features(int n_samples, int initial_models) {
    StackedFeatures sf;
    sf.n_samples = n_samples;
    sf.n_models = 0;
    sf.base_probs = malloc(n_samples * sizeof(double *));
    for (int i = 0; i < n_samples; i++) {
        sf.base_probs[i] = malloc(initial_models * 2 * sizeof(double)); // 2 classes per model
    }
    return sf;
}

void add_base_model_probs(StackedFeatures *sf, double **probs, int model_classes) {
    for (int i = 0; i < sf->n_samples; i++) {
        // Append new probabilities to existing features
        double *new_row = realloc(sf->base_probs[i], (sf->n_models + 1) * model_classes * sizeof(double));
        if (!new_row) {
            fprintf(stderr, "Memory reallocation failed\n");
            exit(EXIT_FAILURE);
        }
        sf->base_probs[i] = new_row;
        memcpy(&sf->base_probs[i][sf->n_models * model_classes], probs[i], model_classes * sizeof(double));
    }
    sf->n_models++;
}

void train_stacking_model(StackedFeatures *sf, int *labels, LogisticRegression *model, int max_iter, double lr) {
    int n_features = sf->n_models * 2 + 1; // 2 classes per model + bias
    model->n_features = n_features;
    model->weights = calloc(n_features, sizeof(double));

    // Gradient descent
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < sf->n_samples; i++) {
            double z = model->weights[0]; // Bias term
            for (int j = 0; j < sf->n_models * 2; j++) {
                z += model->weights[j + 1] * sf->base_probs[i][j];
            }
            double pred = sigmoid(z);
            double error = pred - labels[i];

            // Update weights
            model->weights[0] -= lr * error; // Bias
            for (int j = 0; j < sf->n_models * 2; j++) {
                model->weights[j + 1] -= lr * error * sf->base_probs[i][j];
            }
        }
    }
}

int *predict_stacking(LogisticRegression *model, StackedFeatures *sf) {
    int *preds = malloc(sf->n_samples * sizeof(int));
    for (int i = 0; i < sf->n_samples; i++) {
        double z = model->weights[0]; // Bias
        for (int j = 0; j < model->n_features - 1; j++) {
            z += model->weights[j + 1] * sf->base_probs[i][j];
        }
        preds[i] = (sigmoid(z) > 0.5) ? 1 : 0;
    }
    return preds;
}

void free_stacked_features(StackedFeatures *sf) {
    for (int i = 0; i < sf->n_samples; i++) {
        free(sf->base_probs[i]);
    }
    free(sf->base_probs);
}

void free_logistic_model(LogisticRegression *model) {
    free(model->weights);
}