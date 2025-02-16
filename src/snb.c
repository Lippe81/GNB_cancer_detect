#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "snb.h"
#include "utils.h"

void train_snb(Dataset *data, SNBModel *model) {
    // Initialize model structure
    model->n_classes = 2;  // Binary classification
    model->n_features = data->n_features;
    
    // Allocate memory
    model->means = malloc(2 * sizeof(double*));
    model->variances = malloc(2 * sizeof(double*));
    model->class_priors = malloc(2 * sizeof(double));
    
    for(int c = 0; c < 2; c++) {
        model->means[c] = malloc(data->n_features * sizeof(double));
        model->variances[c] = malloc(data->n_features * sizeof(double));
    }

    // Calculate class priors
    int class_counts[2] = {0};
    for(int i = 0; i < data->n_samples; i++) {
        class_counts[(int)data->labels[i]]++;
    }
    
    model->class_priors[0] = (double)class_counts[0] / data->n_samples;
    model->class_priors[1] = (double)class_counts[1] / data->n_samples;

    // Calculate means and variances with semi-naive assumption
    for(int c = 0; c < 2; c++) {
        for(int f = 0; f < data->n_features; f++) {
            double sum = 0.0;
            double sum_sq = 0.0;
            int count = 0;
            
            for(int i = 0; i < data->n_samples; i++) {
                if(data->labels[i] == c) {
                    double val = data->features[i][f];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
            
            model->means[c][f] = sum / count;
            model->variances[c][f] = (sum_sq - (sum * sum)/count) / (count - 1);
            
            // Add small epsilon to avoid zero variance
            if(model->variances[c][f] < 1e-9) {
                model->variances[c][f] = 1e-9;
            }
        }
    }
}

double** get_snb_probs(SNBModel *model, Dataset *data) {
    double** probs = malloc(data->n_samples * sizeof(double*));
    for(int i = 0; i < data->n_samples; i++) {
        probs[i] = malloc(2 * sizeof(double));
        double log_prob[2] = {log(model->class_priors[0]), log(model->class_priors[1])};
        
        for(int c = 0; c < 2; c++) {
            for(int f = 0; f < model->n_features; f++) {
                double x = data->features[i][f];
                double mean = model->means[c][f];
                double var = model->variances[c][f];
                
                // Semi-naive modification: Consider pairwise feature interactions
                if(f % 2 == 0 && f+1 < model->n_features) {
                    double x2 = data->features[i][f+1];
                    double mean2 = model->means[c][f+1];
                    double var2 = model->variances[c][f+1];
                    
                    // Calculate joint probability for pair of features
                    log_prob[c] += -0.5 * (pow((x - mean), 2)/var + 
                                         pow((x2 - mean2), 2)/var2);
                } else {
                    log_prob[c] += -0.5 * pow((x - mean), 2)/var;
                }
            }
        }
        
        // Convert log probabilities to actual probabilities
        double max_log = fmax(log_prob[0], log_prob[1]);
        probs[i][0] = exp(log_prob[0] - max_log);
        probs[i][1] = exp(log_prob[1] - max_log);
        
        // Normalize
        double sum = probs[i][0] + probs[i][1];
        probs[i][0] /= sum;
        probs[i][1] /= sum;
    }
    return probs;
}

void free_snb_model(SNBModel *model) {
    for(int c = 0; c < 2; c++) {
        free(model->means[c]);
        free(model->variances[c]);
    }
    free(model->means);
    free(model->variances);
    free(model->class_priors);
}