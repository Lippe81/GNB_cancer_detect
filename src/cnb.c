#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnb.h"
#include "utils.h"

#define ALPHA 1.0  // Laplace smoothing

void train_cnb(Dataset *data, CNBModel *model) {
    model->n_classes = 2;
    model->n_features = data->n_features;
    
    // Initialize arrays
    model->class_log_prior = malloc(2 * sizeof(double));
    model->feature_log_prob = malloc(2 * sizeof(double*));
    
    int class_counts[2] = {0};
    double *class_feature_sum = malloc(2 * sizeof(double));
    
    for(int c = 0; c < 2; c++) {
        model->feature_log_prob[c] = malloc(data->n_features * sizeof(double));
        class_feature_sum[c] = 0.0;
    }

    // Calculate class counts and total feature sums
    for(int i = 0; i < data->n_samples; i++) {
        int c = (int)data->labels[i];
        class_counts[c]++;
        for(int f = 0; f < data->n_features; f++) {
            class_feature_sum[c] += data->features[i][f];
        }
    }

    // Calculate log priors
    for(int c = 0; c < 2; c++) {
        model->class_log_prior[c] = log((double)class_counts[c] / data->n_samples);
    }

    // Calculate feature log probabilities with complement
    for(int c = 0; c < 2; c++) {
        int complement_class = 1 - c;
        double complement_sum = class_feature_sum[complement_class] + ALPHA * data->n_features;
        
        for(int f = 0; f < data->n_features; f++) {
            double feature_sum = 0.0;
            for(int i = 0; i < data->n_samples; i++) {
                if((int)data->labels[i] == complement_class) {
                    feature_sum += data->features[i][f];
                }
            }
            model->feature_log_prob[c][f] = log(feature_sum + ALPHA) - log(complement_sum);
        }
    }
    
    free(class_feature_sum);
}

double** get_cnb_probs(CNBModel *model, Dataset *data) {
    double** probs = malloc(data->n_samples * sizeof(double*));
    for(int i = 0; i < data->n_samples; i++) {
        probs[i] = malloc(2 * sizeof(double));
        double log_prob[2] = {model->class_log_prior[0], model->class_log_prior[1]};
        
        for(int c = 0; c < 2; c++) {
            for(int f = 0; f < model->n_features; f++) {
                log_prob[c] += data->features[i][f] * model->feature_log_prob[c][f];
            }
        }
        
        // Softmax
        double max_log = fmax(log_prob[0], log_prob[1]);
        probs[i][0] = exp(log_prob[0] - max_log);
        probs[i][1] = exp(log_prob[1] - max_log);
        double sum = probs[i][0] + probs[i][1];
        probs[i][0] /= sum;
        probs[i][1] /= sum;
    }
    return probs;
}

void free_cnb_model(CNBModel *model) {
    for(int c = 0; c < 2; c++) {
        free(model->feature_log_prob[c]);
    }
    free(model->feature_log_prob);
    free(model->class_log_prior);
}