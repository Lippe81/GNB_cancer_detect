#ifndef STACKING_H
#define STACKING_H

// Structure to hold base model predictions (extensible for future models)
typedef struct {
    double **base_probs; // 2D array [n_samples][n_models * n_classes]
    int n_samples;
    int n_models;        // Current number of models (e.g., 2 for GNB and KDE NB)
} StackedFeatures;

// Logistic regression meta-model
typedef struct {
    double *weights;     // Weight for each feature (including bias)
    int n_features;      // Number of features (n_models * n_classes + 1 for bias)
} LogisticRegression;

// Initialize stacked features (extensible)
StackedFeatures init_stacked_features(int n_samples, int initial_models);

// Add base model predictions to stacked features
void add_base_model_probs(StackedFeatures *sf, double **probs, int model_classes);

// Train logistic regression meta-model
void train_stacking_model(StackedFeatures *sf, int *labels, LogisticRegression *model, int max_iter, double lr);

// Free memory for stacked features and model
void free_stacked_features(StackedFeatures *sf);
void free_logistic_model(LogisticRegression *model);

// Predict using the stacking model
int *predict_stacking(LogisticRegression *model, StackedFeatures *sf);

#endif // STACKING_H