#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "preprocessing.h"
#include "utils.h" // For helper functions

// Compute min and max for each feature in the training set
static void compute_min_max(const Dataset *data, double *min, double *max) {
    for (int j = 0; j < data->n_features; j++) {
        min[j] = data->features[0][j];
        max[j] = data->features[0][j];
        for (int i = 1; i < data->n_samples; i++) {
            if (data->features[i][j] < min[j]) min[j] = data->features[i][j];
            if (data->features[i][j] > max[j]) max[j] = data->features[i][j];
        }
    }
}

// Apply min-max scaling to a dataset using precomputed min/max
static void apply_normalization(Dataset *data, const double *min, const double *max) {
    for (int i = 0; i < data->n_samples; i++) {
        for (int j = 0; j < data->n_features; j++) {
            double range = max[j] - min[j];
            if (range < 1e-9) range = 1.0; // Avoid division by zero
            data->features[i][j] = (data->features[i][j] - min[j]) / range;
        }
    }
}

// Normalize training and test sets using training data's min/max
void normalize(Dataset *train, Dataset *test) {
    double *min = malloc(train->n_features * sizeof(double));
    double *max = malloc(train->n_features * sizeof(double));
    compute_min_max(train, min, max);
    apply_normalization(train, min, max);
    apply_normalization(test, min, max);
    free(min);
    free(max);
}

// Split dataset into training and testing sets (simple random split)
void train_test_split(const Dataset *data, Dataset *train, Dataset *test, float test_size) {
    srand(time(NULL)); // Seed for shuffling

    // Create an array of indices and shuffle them
    int *indices = malloc(data->n_samples * sizeof(int));
    for (int i = 0; i < data->n_samples; i++) indices[i] = i;
    shuffle(indices, data->n_samples);

    // Compute split sizes
    int test_samples = (int)(data->n_samples * test_size);
    int train_samples = data->n_samples - test_samples;

    // Initialize train/test datasets
    train->n_samples = train_samples;
    test->n_samples = test_samples;
    train->n_features = test->n_features = data->n_features;

    // Allocate memory
    train->features = malloc(train_samples * sizeof(double *));
    train->labels = malloc(train_samples * sizeof(int));
    test->features = malloc(test_samples * sizeof(double *));
    test->labels = malloc(test_samples * sizeof(int));

    // Populate train/test sets
    for (int i = 0; i < data->n_samples; i++) {
        if (i < train_samples) {
            train->features[i] = copy_array(data->features[indices[i]], data->n_features);
            train->labels[i] = data->labels[indices[i]];
        } else {
            test->features[i - train_samples] = copy_array(data->features[indices[i]], data->n_features);
            test->labels[i - train_samples] = data->labels[indices[i]];
        }
    }

    free(indices);
}