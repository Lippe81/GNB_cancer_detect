#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_LINE_LENGTH 1024
#define MAX_FEATURES 30
#define EPSILON 1e-9  // To prevent zero variance

typedef struct {
    double mean;
    double variance;
} GaussianParams;

typedef struct {
    GaussianParams features[MAX_FEATURES];
    double prior;
} ClassParams;

void calculate_mean_variance(double *data, int n, GaussianParams *params) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    params->mean = sum / n;

    double variance_sum = 0.0;
    for (int i = 0; i < n; i++) {
        variance_sum += pow(data[i] - params->mean, 2);
    }
    params->variance = variance_sum / n + EPSILON;  // Add epsilon to avoid zero variance
}

double gaussian_probability(double x, GaussianParams *params) {
    double variance = params->variance;
    double exponent = exp(-pow(x - params->mean, 2) / (2 * variance));
    return (1.0 / sqrt(2 * M_PI * variance)) * exponent;
}

void train_gnb(double **X, char *y, int n_samples, int n_features, ClassParams *class_M, ClassParams *class_B) {
    int count_M = 0, count_B = 0;

    for (int i = 0; i < n_samples; i++) {
        if (y[i] == 'M') {
            count_M++;
        } else {
            count_B++;
        }
    }

    class_M->prior = (double)count_M / n_samples;
    class_B->prior = (double)count_B / n_samples;

    for (int j = 0; j < n_features; j++) {
        double *data_M = malloc(count_M * sizeof(double));
        double *data_B = malloc(count_B * sizeof(double));
        int idx_M = 0, idx_B = 0;

        for (int i = 0; i < n_samples; i++) {
            if (y[i] == 'M') {
                data_M[idx_M++] = X[i][j];
            } else {
                data_B[idx_B++] = X[i][j];
            }
        }

        calculate_mean_variance(data_M, count_M, &class_M->features[j]);
        calculate_mean_variance(data_B, count_B, &class_B->features[j]);

        free(data_M);
        free(data_B);
    }
}

char predict_gnb(double *x, int n_features, ClassParams *class_M, ClassParams *class_B) {
    double log_prob_M = log(class_M->prior);
    double log_prob_B = log(class_B->prior);

    for (int j = 0; j < n_features; j++) {
        double prob_M = gaussian_probability(x[j], &class_M->features[j]);
        double prob_B = gaussian_probability(x[j], &class_B->features[j]);
        log_prob_M += log(prob_M + EPSILON);  // Avoid log(0)
        log_prob_B += log(prob_B + EPSILON);
    }

    return log_prob_M > log_prob_B ? 'M' : 'B';
}

void read_csv(const char *filename, double ***X, char **y, int *n_samples, int *n_features) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;
    int feature_count = 0;

    // Read header to determine feature count
    if (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        while (token != NULL) {
            feature_count++;
            token = strtok(NULL, ",");
        }
        *n_features = feature_count - 2;  // Exclude id and diagnosis columns
    }

    // Count number of samples (excluding header)
    while (fgets(line, sizeof(line), file)) {
        sample_count++;
    }
    *n_samples = sample_count;

    // Allocate memory
    *X = malloc(*n_samples * sizeof(double *));
    *y = malloc(*n_samples * sizeof(char));

    // Reset file pointer to read data
    fseek(file, 0, SEEK_SET);
    fgets(line, sizeof(line), file);  // Skip header

    int sample_idx = 0;
    while (fgets(line, sizeof(line), file)) {
        (*X)[sample_idx] = malloc(*n_features * sizeof(double));
        char *token = strtok(line, ",");  // Skip id
        token = strtok(NULL, ",");        // Get diagnosis
        (*y)[sample_idx] = token[0];

        // Read features
        for (int j = 0; j < *n_features; j++) {
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "Error reading features for sample %d\n", sample_idx);
                exit(EXIT_FAILURE);
            }
            (*X)[sample_idx][j] = atof(token);
        }
        sample_idx++;
    }

    fclose(file);
}

void evaluate_model(double **X, char *y, int n_samples, int n_features, ClassParams *class_M, ClassParams *class_B) {
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < n_samples; i++) {
        char prediction = predict_gnb(X[i], n_features, class_M, class_B);
        if (prediction == 'M' && y[i] == 'M') tp++;
        else if (prediction == 'B' && y[i] == 'B') tn++;
        else if (prediction == 'M' && y[i] == 'B') fp++;
        else if (prediction == 'B' && y[i] == 'M') fn++;
    }

    double accuracy = (double)(tp + tn) / n_samples;
    double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
    double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
    double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    double error = 1 - accuracy;

    printf("Confusion Matrix:\n");
    printf("TP: %d, TN: %d, FP: %d, FN: %d\n", tp, tn, fp, fn);
    printf("Accuracy: %.2f%%\n", accuracy * 100);
    printf("Precision: %.2f%%\n", precision * 100);
    printf("Recall: %.2f%%\n", recall * 100);
    printf("F1 Score: %.2f%%\n", f1_score * 100);
    printf("Prediction Error: %.2f%%\n", error * 100);
}

int main() {
    const char *csv_filename = "breast_cancer.csv";
    double **X;
    char *y;
    int n_samples, n_features;

    FILE *file = fopen(csv_filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file: %s\n", csv_filename);
        return EXIT_FAILURE;
    }
    fclose(file);

    read_csv(csv_filename, &X, &y, &n_samples, &n_features);

    ClassParams class_M, class_B;
    train_gnb(X, y, n_samples, n_features, &class_M, &class_B);

    evaluate_model(X, y, n_samples, n_features, &class_M, &class_B);

    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
    }
    free(X);
    free(y);

    return 0;
}