#include <stdio.h>
#include <stdlib.h>
#include "data_loader.h"
#include "preprocessing.h"
#include "gnb.h"
#include "kde_nb.h"
#include "stacking.h"
#include "evaluation.h"
#include "utils.h" 


int main() {
    // Load and preprocess data
    Dataset data, train, test;
    load_csv("data/breast-cancer.csv", &data);
    train_test_split(&data, &train, &test, 0.3);
    normalize(&train, &test);

    // Train base models
    GNBModel gnb_model;
    train_gnb(&train, &gnb_model);
    KDEModel kde_model;
    train_kde_nb(&train, &kde_model);

    // Generate base model probabilities for stacking
    StackedFeatures sf = init_stacked_features(test.n_samples, 2);
    double **gnb_probs = get_gnb_probs(&gnb_model, &test);
    add_base_model_probs(&sf, gnb_probs, 2);
    double **kde_probs = get_kde_probs(&kde_model, &test); // Use the new function
    add_base_model_probs(&sf, kde_probs, 2);

    // Train meta-model
    LogisticRegression meta_model;
    train_stacking_model(&sf, test.labels, &meta_model, 1000, 0.01);

    // Evaluate
    int *final_preds = predict_stacking(&meta_model, &sf);
    ConfusionMatrix cm = compute_confusion_matrix(test.labels, final_preds, test.n_samples);
    printf("Accuracy: %.2f%%\n", accuracy(&cm) * 100);
    printf("F1 Score: %.2f%%\n", f1_score(&cm) * 100);

    // Cleanup
    free_stacked_features(&sf);
    free_logistic_model(&meta_model);
    free(final_preds);
    free_2d_array(gnb_probs, test.n_samples);
    free_2d_array(kde_probs, test.n_samples); // Free KDE probabilities
    free_dataset(&data);
    free_dataset(&train);
    free_dataset(&test);
    return 0;
}