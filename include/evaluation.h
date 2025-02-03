#ifndef EVALUATION_H
#define EVALUATION_H

typedef struct {
    int tp; // True positives
    int fp; // False positives
    int tn; // True negatives
    int fn; // False negatives
} ConfusionMatrix;

ConfusionMatrix compute_confusion_matrix(const int *y_true, const int *y_pred, int n_samples);
double accuracy(const ConfusionMatrix *cm);
double precision(const ConfusionMatrix *cm);
double recall(const ConfusionMatrix *cm);
double f1_score(const ConfusionMatrix *cm);
double prediction_error(const ConfusionMatrix *cm);

#endif // EVALUATION_H